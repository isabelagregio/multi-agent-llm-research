#!/usr/bin/env python3
"""
Experiment H -- Risk-Aware Interaction Router (RAIR)
====================================================

Goal
----
Learn an interpretable controller that decides:

    1) whether to STOP after the initial single-model solution, or activate one of
       three interaction protocols:
           - SELF_REFINE
           - STUDENT_CRITIC
           - DEBATE

    2) after a protocol is selected, whether to STOP or CONTINUE within that
       protocol up to a fixed maximum checkpoint (R2 by default).

Important methodological constraint
-----------------------------------
Experiment B contains separate trajectories for Self-Refine, Student-Critic and
Debate. It does NOT contain counterfactual trajectories for switching protocols
mid-run (e.g., Self-Refine -> Debate). Therefore this experiment performs:

    protocol selection at the common initial state
        + protocol-specific adaptive stopping.

This is fully evaluable from the existing traces without inventing unseen states.

No benchmark difficulty label or ground-truth correctness is used as a runtime
feature. Gold correctness is used only offline to construct HELP/HARM labels and
for final evaluation.

Evaluation
----------
Five-fold outer cross-fitting by question_id. For each outer fold:
    train: fit HELP/HARM classifiers
    dev:   select policy hyperparameters under an accuracy constraint
    test:  evaluate the frozen policy on unseen question IDs

All stochastic repetitions of a question stay in the same split.

Main methods reported
---------------------
    single
    fixed_self_r1
    fixed_self_r2
    fixed_critic_r1
    fixed_critic_r2
    fixed_debate_r0
    fixed_debate_r1
    fixed_debate_r2
    utility_only_router
    risk_aware_selector
    rair

RAIR score for a candidate action a:

    score(a | s) = P(help | s,a)
                   - lambda_harm * P(harm | s,a)
                   - lambda_cost * E[incremental_tokens(a)] / 1000

The action is taken only if the score exceeds a threshold and its predicted harm
is below a cap. Otherwise the controller stops.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import io
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from math_verify import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        parse as math_parse,
        verify as math_verify,
    )
except ImportError as exc:
    raise SystemExit(
        "Install math-verify with: pip install 'math-verify[antlr4_13_2]'"
    ) from exc


# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(os.environ.get("PROJECT_ROOT", "/lustre06/project/6087921/isa1909"))
B_RAW = ROOT / "results" / "experiment_7_round_scaling_difficulty" / "raw_results.jsonl"
OUT = ROOT / "results" / "experiment_H_risk_aware_interaction_router"

N_FOLDS = 5
DEV_FRAC = 0.25
SEED = 20261106
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261107
MAX_PROTOCOL_ROUND = 2
ACCURACY_TOLERANCE = 0.005  # 0.5 percentage point

PROTOCOLS = ("self_refine", "student_critic", "debate")

# Hyperparameter grid chosen on dev only.
HARM_WEIGHTS = (0.5, 1.0, 2.0)
COST_WEIGHTS = (0.0, 0.05, 0.10)
SCORE_THRESHOLDS = (-0.10, 0.0, 0.05, 0.10)
HARM_CAPS = (0.20, 0.40, 1.00)

# Utility-only ablation: ignores P(harm).
UTILITY_COST_WEIGHTS = COST_WEIGHTS
UTILITY_THRESHOLDS = SCORE_THRESHOLDS

INITIAL_FEATURES = [
    "log_text_chars",
    "log_total_tokens",
    "log_answer_chars",
    "num_lines_log",
    "digit_fraction",
    "math_symbol_fraction",
    "has_boxed",
    "has_fraction",
    "has_sqrt",
    "has_equation_like",
]

SELF_FEATURES = [
    "round_index",
    "answer_stable",
    "answer_changed",
    "log_output_chars",
    "log_output_tokens",
    "log_aux_chars",
    "log_aux_tokens",
    "answer_length_change",
    "output_length_change",
]

CRITIC_FEATURES = SELF_FEATURES + [
    "verdict_accept",
    "verdict_reject",
    "verdict_unknown",
]

DEBATE_FEATURES = [
    "round_index",
    "current_disagreement",
    "current_consensus",
    "new_consensus",
    "agent_a_stable",
    "agent_b_stable",
    "both_stable",
    "both_changed",
    "one_changed",
    "a_adopts_prev_b",
    "b_adopts_prev_a",
    "answer_length_rel_diff",
    "output_tokens_rel_diff",
]

METHODS = [
    "single",
    "fixed_self_r1",
    "fixed_self_r2",
    "fixed_critic_r1",
    "fixed_critic_r2",
    "fixed_debate_r0",
    "fixed_debate_r1",
    "fixed_debate_r2",
    "utility_only_router",
    "risk_aware_selector",
    "rair",
]


# =============================================================================
# Basic I/O
# =============================================================================


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def save_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Trace helpers
# =============================================================================


def normalize_answer(value) -> str:
    """Cheap normalization used before symbolic verification.

    This intentionally handles only surface-form equivalence. More difficult
    mathematical equivalence is delegated to math_verify below.
    """
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = s.replace("$", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\boxed\s*\{(.*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    return s


def parse_math_value(value):
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    if not ((s.startswith("$") and s.endswith("$")) or "\\boxed" in s):
        s = f"${s}$"
    return math_parse(
        s,
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
    )


# math_verify is robust but can be expensive on a small number of pathological
# MATH expressions. H reuses the same answer pairs many times while tuning
# policies, so caching is critical. We also avoid symbolic verification for
# unusually long final-answer strings; those are treated conservatively as
# non-equivalent unless their normalized forms already match.
MAX_SYMBOLIC_EQUIV_CHARS = 160


@lru_cache(maxsize=100_000)
def _equivalent_cached(a: str, b: str) -> bool:
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return False
    if na == nb:
        return True

    if max(len(a), len(b)) > MAX_SYMBOLIC_EQUIV_CHARS:
        return False

    try:
        pa, pb = parse_math_value(a), parse_math_value(b)
        if pa and pb:
            # math_verify may print timeout diagnostics internally. They are not
            # actionable here because a failed/timeout comparison simply means
            # we conservatively treat the answers as different runtime states.
            sink = io.StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                return bool(math_verify(pa, pb))
    except Exception:
        pass
    return False


def equivalent(a, b) -> bool:
    if a is None or b is None:
        return False
    sa, sb = str(a).strip(), str(b).strip()
    if not sa or not sb:
        return False

    # Canonicalize argument order so equivalent(a,b) and equivalent(b,a) share
    # the same cache entry.
    if sa <= sb:
        return _equivalent_cached(sa, sb)
    return _equivalent_cached(sb, sa)


def obj_text(obj) -> str:
    if isinstance(obj, dict):
        return str(obj.get("text") or "")
    return str(obj or "")


def obj_answer(obj) -> str:
    if isinstance(obj, dict):
        if obj.get("answer") is not None:
            return str(obj["answer"])
        return str(obj.get("text") or "")
    return str(obj or "")


def obj_tokens(obj) -> int:
    if isinstance(obj, dict):
        return int(obj.get("total_tokens") or 0)
    return 0


def safe_log1p(x: float) -> float:
    return float(math.log1p(max(0.0, float(x))))


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max((abs(a) + abs(b)) / 2.0, 1.0)


def single_output(row: dict):
    return row["single"]["output"]


def single_correct(row: dict) -> bool:
    return bool(row["single"]["correct"])


def single_tokens(row: dict) -> float:
    return float(obj_tokens(single_output(row)))


def protocol_state(row: dict, protocol: str, round_id: int) -> dict:
    return row[protocol][str(round_id)]


def protocol_correct(row: dict, protocol: str, round_id: int) -> bool:
    return bool(protocol_state(row, protocol, round_id)["correct"])


def protocol_tokens(row: dict, protocol: str, round_id: int) -> float:
    state = protocol_state(row, protocol, round_id)
    if protocol == "debate":
        return float(state["cumulative_protocol_tokens"])
    return float(state["cumulative_tokens"])


def total_runtime_tokens(row: dict, protocol: str, round_id: int) -> float:
    """Total cost of the evaluated controller path.

    Self-Refine and Student-Critic trajectories in Experiment B reuse the same
    initial Single solution, so their cumulative token counts already include S0.

    Debate was generated as a separate protocol path. Because RAIR first observes
    the Single solution before deciding to launch Debate, we conservatively add the
    initial Single cost to the stored Debate cost.
    """
    base = protocol_tokens(row, protocol, round_id)
    if protocol == "debate":
        return single_tokens(row) + base
    return base


def initial_action_round(protocol: str) -> int:
    # Self-Refine/Student-Critic R0 == Single, so activation means taking R1.
    # Debate R0 is already an independent multi-agent inference + judge checkpoint.
    return 0 if protocol == "debate" else 1


def initial_action_outcome(row: dict, protocol: str) -> Tuple[bool, float, int]:
    r = initial_action_round(protocol)
    return protocol_correct(row, protocol, r), total_runtime_tokens(row, protocol, r), r


def next_protocol_rounds(protocol: str) -> List[int]:
    # Return current rounds from which a CONTINUE decision is possible.
    if protocol == "debate":
        return list(range(0, MAX_PROTOCOL_ROUND))
    return list(range(1, MAX_PROTOCOL_ROUND))


# =============================================================================
# Feature construction
# =============================================================================


def initial_features(row: dict) -> np.ndarray:
    out = single_output(row)
    text = obj_text(out)
    answer = obj_answer(out)
    n = max(len(text), 1)
    digit_fraction = sum(c.isdigit() for c in text) / n
    math_symbols = set("=+-*/^_{}[]()<>\\")
    math_symbol_fraction = sum(c in math_symbols for c in text) / n
    vals = [
        safe_log1p(len(text)),
        safe_log1p(obj_tokens(out)),
        safe_log1p(len(answer)),
        safe_log1p(text.count("\n") + 1),
        digit_fraction,
        math_symbol_fraction,
        float("\\boxed" in text),
        float("\\frac" in text),
        float("\\sqrt" in text),
        float("=" in text or "\\implies" in text or "\\Rightarrow" in text),
    ]
    return np.asarray(vals, dtype=float)


def self_or_critic_features(row: dict, protocol: str, round_id: int) -> np.ndarray:
    current = protocol_state(row, protocol, round_id)
    previous = protocol_state(row, protocol, round_id - 1)

    cur_out = current["output"]
    prev_out = previous["output"]
    cur_answer = obj_answer(cur_out)
    prev_answer = obj_answer(prev_out)
    cur_text = obj_text(cur_out)
    prev_text = obj_text(prev_out)

    stable = equivalent(cur_answer, prev_answer)
    aux = current.get("feedback") if protocol == "self_refine" else current.get("critique")
    aux_text = obj_text(aux)

    vals = [
        float(round_id),
        float(stable),
        float(not stable),
        safe_log1p(len(cur_text)),
        safe_log1p(obj_tokens(cur_out)),
        safe_log1p(len(aux_text)),
        safe_log1p(obj_tokens(aux)),
        rel_diff(len(cur_answer), len(prev_answer)),
        rel_diff(len(cur_text), len(prev_text)),
    ]

    if protocol == "student_critic":
        verdict = str(current.get("verdict") or "").upper().strip()
        vals.extend(
            [
                float(verdict == "ACCEPT"),
                float(verdict == "REJECT"),
                float(verdict not in {"ACCEPT", "REJECT"}),
            ]
        )

    return np.asarray(vals, dtype=float)


def debate_agents(row: dict, round_id: int) -> Tuple[str, str]:
    state = protocol_state(row, "debate", round_id)
    return obj_answer(state["agent_a"]), obj_answer(state["agent_b"])


def debate_features(row: dict, round_id: int) -> np.ndarray:
    state = protocol_state(row, "debate", round_id)
    a, b = debate_agents(row, round_id)
    disagreement = float(not equivalent(a, b))
    consensus = 1.0 - disagreement

    a_text = obj_text(state["agent_a"])
    b_text = obj_text(state["agent_b"])
    a_tok = obj_tokens(state["agent_a"])
    b_tok = obj_tokens(state["agent_b"])

    if round_id == 0:
        return np.asarray(
            [
                0.0,
                disagreement,
                consensus,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                rel_diff(len(a_text), len(b_text)),
                rel_diff(a_tok, b_tok),
            ],
            dtype=float,
        )

    pa, pb = debate_agents(row, round_id - 1)
    a_stable = equivalent(a, pa)
    b_stable = equivalent(b, pb)
    prev_disagreement = not equivalent(pa, pb)

    return np.asarray(
        [
            float(round_id),
            disagreement,
            consensus,
            float(prev_disagreement and not bool(disagreement)),
            float(a_stable),
            float(b_stable),
            float(a_stable and b_stable),
            float((not a_stable) and (not b_stable)),
            float(a_stable != b_stable),
            float(equivalent(a, pb) and not a_stable),
            float(equivalent(b, pa) and not b_stable),
            rel_diff(len(a_text), len(b_text)),
            rel_diff(a_tok, b_tok),
        ],
        dtype=float,
    )


def protocol_features(row: dict, protocol: str, round_id: int) -> np.ndarray:
    if protocol == "debate":
        return debate_features(row, round_id)
    return self_or_critic_features(row, protocol, round_id)


def feature_names_for_protocol(protocol: str) -> List[str]:
    if protocol == "self_refine":
        return SELF_FEATURES
    if protocol == "student_critic":
        return CRITIC_FEATURES
    if protocol == "debate":
        return DEBATE_FEATURES
    raise KeyError(protocol)


# =============================================================================
# Training records
# =============================================================================


def transition_labels(before_correct: bool, after_correct: bool) -> Tuple[int, int]:
    help_label = int((not before_correct) and after_correct)
    harm_label = int(before_correct and (not after_correct))
    return help_label, harm_label


def build_initial_action_records(raw: Sequence[dict]) -> Dict[str, List[dict]]:
    records = {p: [] for p in PROTOCOLS}
    for row in raw:
        before = single_correct(row)
        base_tokens = single_tokens(row)
        x = initial_features(row)
        for protocol in PROTOCOLS:
            after, total_tokens, r = initial_action_outcome(row, protocol)
            help_label, harm_label = transition_labels(before, after)
            records[protocol].append(
                {
                    "question_id": int(row["question_id"]),
                    "repetition": int(row["repetition"]),
                    "protocol": protocol,
                    "round": r,
                    "features": x,
                    "help": help_label,
                    "harm": harm_label,
                    "before_correct": before,
                    "after_correct": after,
                    "incremental_tokens": max(0.0, total_tokens - base_tokens),
                }
            )
    return records


def build_continuation_records(raw: Sequence[dict]) -> Dict[str, List[dict]]:
    records = {p: [] for p in PROTOCOLS}
    for row in raw:
        for protocol in PROTOCOLS:
            for r in next_protocol_rounds(protocol):
                before = protocol_correct(row, protocol, r)
                after = protocol_correct(row, protocol, r + 1)
                help_label, harm_label = transition_labels(before, after)
                current_total = total_runtime_tokens(row, protocol, r)
                next_total = total_runtime_tokens(row, protocol, r + 1)
                records[protocol].append(
                    {
                        "question_id": int(row["question_id"]),
                        "repetition": int(row["repetition"]),
                        "protocol": protocol,
                        "round": r,
                        "features": protocol_features(row, protocol, r),
                        "help": help_label,
                        "harm": harm_label,
                        "before_correct": before,
                        "after_correct": after,
                        "incremental_tokens": max(0.0, next_total - current_total),
                    }
                )
    return records


# =============================================================================
# Models
# =============================================================================


class ConstantProbabilityModel:
    """Fallback for a label with only one class in a training fold."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X):
        X = np.asarray(X)
        p = np.full(len(X), self.p, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class ActionModels:
    help_model: object
    harm_model: object
    mean_incremental_tokens: float



def fit_binary_model(records: Sequence[dict], label: str):
    X = np.vstack([r["features"] for r in records])
    y = np.asarray([r[label] for r in records], dtype=int)
    unique = np.unique(y)
    if len(unique) < 2:
        return ConstantProbabilityModel(float(np.mean(y)))
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def fit_action_models(records: Sequence[dict]) -> ActionModels:
    return ActionModels(
        help_model=fit_binary_model(records, "help"),
        harm_model=fit_binary_model(records, "harm"),
        mean_incremental_tokens=float(mean(r["incremental_tokens"] for r in records)),
    )


def predict_prob(model, x: np.ndarray) -> float:
    return float(model.predict_proba(x.reshape(1, -1))[0, 1])


def classifier_metrics(model, records: Sequence[dict], label: str) -> dict:
    if not records:
        return {"roc_auc": None, "average_precision": None, "prevalence": None, "n": 0}
    X = np.vstack([r["features"] for r in records])
    y = np.asarray([r[label] for r in records], dtype=int)
    p = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "average_precision": float(average_precision_score(y, p)) if np.any(y == 1) else None,
        "prevalence": float(np.mean(y)),
        "n": int(len(y)),
    }


def extract_coefficients(model, feature_names: Sequence[str]) -> Optional[List[float]]:
    if isinstance(model, Pipeline) and "logreg" in model.named_steps:
        return [float(v) for v in model.named_steps["logreg"].coef_[0]]
    return None


# =============================================================================
# Splits
# =============================================================================


def make_outer_folds(raw: Sequence[dict], n_folds: int, seed: int) -> List[List[int]]:
    # Stratification by MATH level is only for balanced evaluation splits.
    # Level is never used as a model/runtime feature.
    q_level = {int(r["question_id"]): int(r["level"]) for r in raw}
    rng = random.Random(seed)
    folds = [[] for _ in range(n_folds)]
    for level in sorted(set(q_level.values())):
        qids = [q for q, l in q_level.items() if l == level]
        rng.shuffle(qids)
        for i, qid in enumerate(qids):
            folds[i % n_folds].append(qid)
    return [sorted(f) for f in folds]


def split_train_dev(
    qids: Sequence[int], raw: Sequence[dict], dev_frac: float, seed: int
) -> Tuple[List[int], List[int]]:
    q_level = {int(r["question_id"]): int(r["level"]) for r in raw}
    by_level = defaultdict(list)
    for qid in qids:
        by_level[q_level[qid]].append(qid)

    rng = random.Random(seed)
    train, dev = [], []
    for level, ids in sorted(by_level.items()):
        ids = ids[:]
        rng.shuffle(ids)
        n_dev = max(1, int(round(len(ids) * dev_frac)))
        dev.extend(ids[:n_dev])
        train.extend(ids[n_dev:])
    return sorted(train), sorted(dev)


# =============================================================================
# Policy execution
# =============================================================================


@dataclass(frozen=True)
class PolicyParams:
    harm_weight: float
    cost_weight: float
    score_threshold: float
    harm_cap: float


@dataclass(frozen=True)
class UtilityParams:
    cost_weight: float
    score_threshold: float



def action_score(
    p_help: float,
    p_harm: float,
    expected_incremental_tokens: float,
    params: PolicyParams,
) -> float:
    return (
        p_help
        - params.harm_weight * p_harm
        - params.cost_weight * (expected_incremental_tokens / 1000.0)
    )


def utility_score(
    p_help: float,
    expected_incremental_tokens: float,
    params: UtilityParams,
) -> float:
    return p_help - params.cost_weight * (expected_incremental_tokens / 1000.0)


def make_result(
    *,
    correct: bool,
    tokens: float,
    protocol: str,
    round_id: int,
    helpful_exposures: int,
    harmful_exposures: int,
    stop_reason: str,
    decisions: Optional[list] = None,
) -> dict:
    return {
        "correct": bool(correct),
        "tokens": float(tokens),
        "protocol": protocol,
        "round": int(round_id),
        "helpful_exposures": int(helpful_exposures),
        "harmful_exposures": int(harmful_exposures),
        "stop_reason": stop_reason,
        "decisions": decisions or [],
    }


def fixed_single(row: dict) -> dict:
    return make_result(
        correct=single_correct(row),
        tokens=single_tokens(row),
        protocol="single",
        round_id=0,
        helpful_exposures=0,
        harmful_exposures=0,
        stop_reason="fixed_single",
    )


def native_path_exposures(row: dict, protocol: str, round_id: int) -> Tuple[int, int]:
    """Helpful/harmful transitions actually taken by a fixed protocol path."""
    helpful = harmful = 0
    start = 0
    for r in range(start, round_id):
        before = protocol_correct(row, protocol, r)
        after = protocol_correct(row, protocol, r + 1)
        h, d = transition_labels(before, after)
        helpful += h
        harmful += d
    return helpful, harmful


def fixed_protocol(row: dict, protocol: str, round_id: int) -> dict:
    helpful, harmful = native_path_exposures(row, protocol, round_id)
    return make_result(
        correct=protocol_correct(row, protocol, round_id),
        tokens=protocol_tokens(row, protocol, round_id),
        protocol=protocol,
        round_id=round_id,
        helpful_exposures=helpful,
        harmful_exposures=harmful,
        stop_reason=f"fixed_{protocol}_r{round_id}",
    )


def choose_initial_action(
    row: dict,
    initial_models: Mapping[str, ActionModels],
    params,
    utility_only: bool,
) -> Tuple[str, list]:
    x = initial_features(row)
    candidates = []
    for protocol in PROTOCOLS:
        m = initial_models[protocol]
        p_help = predict_prob(m.help_model, x)
        p_harm = predict_prob(m.harm_model, x)
        if utility_only:
            score = utility_score(p_help, m.mean_incremental_tokens, params)
            eligible = score >= params.score_threshold
        else:
            score = action_score(p_help, p_harm, m.mean_incremental_tokens, params)
            eligible = score >= params.score_threshold and p_harm <= params.harm_cap
        candidates.append(
            {
                "protocol": protocol,
                "p_help": p_help,
                "p_harm": p_harm,
                "expected_incremental_tokens": m.mean_incremental_tokens,
                "score": score,
                "eligible": bool(eligible),
            }
        )

    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        return "stop", candidates

    # Score first, then lower expected cost, then lower harm.
    best = max(
        eligible,
        key=lambda c: (c["score"], -c["expected_incremental_tokens"], -c["p_harm"]),
    )
    return best["protocol"], candidates


def should_continue(
    row: dict,
    protocol: str,
    round_id: int,
    continuation_models: Mapping[str, ActionModels],
    params,
    utility_only: bool,
) -> Tuple[bool, dict]:
    x = protocol_features(row, protocol, round_id)
    m = continuation_models[protocol]
    p_help = predict_prob(m.help_model, x)
    p_harm = predict_prob(m.harm_model, x)
    if utility_only:
        score = utility_score(p_help, m.mean_incremental_tokens, params)
        go = score >= params.score_threshold
    else:
        score = action_score(p_help, p_harm, m.mean_incremental_tokens, params)
        go = score >= params.score_threshold and p_harm <= params.harm_cap
    return bool(go), {
        "stage": "continue",
        "protocol": protocol,
        "round": round_id,
        "p_help": p_help,
        "p_harm": p_harm,
        "expected_incremental_tokens": m.mean_incremental_tokens,
        "score": score,
        "continue": bool(go),
    }


def run_router(
    row: dict,
    initial_models: Mapping[str, ActionModels],
    continuation_models: Mapping[str, ActionModels],
    params,
    *,
    utility_only: bool,
    adaptive_depth: bool,
) -> dict:
    action, candidates = choose_initial_action(
        row, initial_models, params, utility_only=utility_only
    )
    decisions = [{"stage": "select_protocol", "candidates": candidates, "chosen": action}]

    if action == "stop":
        z = fixed_single(row)
        z["stop_reason"] = "router_stop_initial"
        z["decisions"] = decisions
        return z

    protocol = action
    r = initial_action_round(protocol)
    before = single_correct(row)
    after = protocol_correct(row, protocol, r)
    helpful, harmful = transition_labels(before, after)

    if not adaptive_depth:
        return make_result(
            correct=after,
            tokens=total_runtime_tokens(row, protocol, r),
            protocol=protocol,
            round_id=r,
            helpful_exposures=helpful,
            harmful_exposures=harmful,
            stop_reason=f"selector_stop_{protocol}_r{r}",
            decisions=decisions,
        )

    while r < MAX_PROTOCOL_ROUND:
        # For Self-Refine and Student-Critic, R0 is the common Single state and
        # activation already moves to R1. Their first continuation decision is R1.
        if protocol != "debate" and r < 1:
            break

        go, info = should_continue(
            row,
            protocol,
            r,
            continuation_models,
            params,
            utility_only=utility_only,
        )
        decisions.append(info)
        if not go:
            break

        before = protocol_correct(row, protocol, r)
        after = protocol_correct(row, protocol, r + 1)
        h, d = transition_labels(before, after)
        helpful += h
        harmful += d
        r += 1

    return make_result(
        correct=protocol_correct(row, protocol, r),
        tokens=total_runtime_tokens(row, protocol, r),
        protocol=protocol,
        round_id=r,
        helpful_exposures=helpful,
        harmful_exposures=harmful,
        stop_reason=f"router_stop_{protocol}_r{r}",
        decisions=decisions,
    )


# =============================================================================
# Model fitting and dev selection
# =============================================================================


def filter_records(records: Sequence[dict], qids: set) -> List[dict]:
    return [r for r in records if r["question_id"] in qids]


def fit_model_bundle(
    initial_records: Mapping[str, Sequence[dict]],
    continuation_records: Mapping[str, Sequence[dict]],
    train_qids: set,
) -> Tuple[Dict[str, ActionModels], Dict[str, ActionModels]]:
    initial_models = {}
    continuation_models = {}
    for protocol in PROTOCOLS:
        ir = filter_records(initial_records[protocol], train_qids)
        cr = filter_records(continuation_records[protocol], train_qids)
        if not ir:
            raise ValueError(f"No initial training records for {protocol}")
        if not cr:
            raise ValueError(f"No continuation training records for {protocol}")
        initial_models[protocol] = fit_action_models(ir)
        continuation_models[protocol] = fit_action_models(cr)
    return initial_models, continuation_models


def policy_metrics(results: Sequence[dict]) -> dict:
    return {
        "accuracy": mean(float(r["correct"]) for r in results),
        "mean_tokens": mean(r["tokens"] for r in results),
        "mean_round": mean(r["round"] for r in results),
        "harmful_exposure_rate": mean(r["harmful_exposures"] for r in results),
        "helpful_exposure_rate": mean(r["helpful_exposures"] for r in results),
    }


def evaluate_router(
    rows: Sequence[dict],
    initial_models: Mapping[str, ActionModels],
    continuation_models: Mapping[str, ActionModels],
    params,
    *,
    utility_only: bool,
    adaptive_depth: bool,
) -> dict:
    results = [
        run_router(
            row,
            initial_models,
            continuation_models,
            params,
            utility_only=utility_only,
            adaptive_depth=adaptive_depth,
        )
        for row in rows
    ]
    return policy_metrics(results)


def best_fixed_accuracy(rows: Sequence[dict]) -> float:
    candidates = []
    for protocol, rounds in (
        ("self_refine", [1, 2]),
        ("student_critic", [1, 2]),
        ("debate", [0, 1, 2]),
    ):
        for r in rounds:
            candidates.append(mean(float(protocol_correct(x, protocol, r)) for x in rows))
    candidates.append(mean(float(single_correct(x)) for x in rows))
    return max(candidates)


def choose_risk_policy(
    dev_rows: Sequence[dict],
    initial_models: Mapping[str, ActionModels],
    continuation_models: Mapping[str, ActionModels],
    *,
    adaptive_depth: bool,
) -> dict:
    target = best_fixed_accuracy(dev_rows) - ACCURACY_TOLERANCE
    candidates = []
    for hw in HARM_WEIGHTS:
        for cw in COST_WEIGHTS:
            for st in SCORE_THRESHOLDS:
                for hc in HARM_CAPS:
                    p = PolicyParams(hw, cw, st, hc)
                    m = evaluate_router(
                        dev_rows,
                        initial_models,
                        continuation_models,
                        p,
                        utility_only=False,
                        adaptive_depth=adaptive_depth,
                    )
                    candidates.append({"params": p, **m})

    qualified = [c for c in candidates if c["accuracy"] >= target]
    if qualified:
        best = min(
            qualified,
            key=lambda c: (
                c["mean_tokens"],
                c["harmful_exposure_rate"],
                -c["accuracy"],
            ),
        )
        rule = "accuracy_constrained_min_cost"
    else:
        best = max(
            candidates,
            key=lambda c: (
                c["accuracy"],
                -c["harmful_exposure_rate"],
                -c["mean_tokens"],
            ),
        )
        rule = "fallback_best_accuracy"

    return {
        **{k: v for k, v in best.items() if k != "params"},
        "params": best["params"],
        "selection_rule": rule,
        "fixed_accuracy_target": target,
    }


def choose_utility_policy(
    dev_rows: Sequence[dict],
    initial_models: Mapping[str, ActionModels],
    continuation_models: Mapping[str, ActionModels],
) -> dict:
    target = best_fixed_accuracy(dev_rows) - ACCURACY_TOLERANCE
    candidates = []
    for cw in UTILITY_COST_WEIGHTS:
        for st in UTILITY_THRESHOLDS:
            p = UtilityParams(cw, st)
            m = evaluate_router(
                dev_rows,
                initial_models,
                continuation_models,
                p,
                utility_only=True,
                adaptive_depth=True,
            )
            candidates.append({"params": p, **m})

    qualified = [c for c in candidates if c["accuracy"] >= target]
    if qualified:
        best = min(
            qualified,
            key=lambda c: (c["mean_tokens"], -c["accuracy"]),
        )
        rule = "accuracy_constrained_min_cost"
    else:
        best = max(candidates, key=lambda c: (c["accuracy"], -c["mean_tokens"]))
        rule = "fallback_best_accuracy"

    return {
        **{k: v for k, v in best.items() if k != "params"},
        "params": best["params"],
        "selection_rule": rule,
        "fixed_accuracy_target": target,
    }


# =============================================================================
# Cross-fitting
# =============================================================================


def crossfit(raw: Sequence[dict]):
    initial_records = build_initial_action_records(raw)
    continuation_records = build_continuation_records(raw)
    folds = make_outer_folds(raw, N_FOLDS, SEED)
    all_qids = sorted({int(r["question_id"]) for r in raw})

    predictions = []
    fold_rows = []
    classifier_rows = []
    coefficient_rows = []

    for fold_idx, test_ids in enumerate(folds, 1):
        test_set = set(test_ids)
        remaining = [q for q in all_qids if q not in test_set]
        train_ids, dev_ids = split_train_dev(
            remaining, raw, DEV_FRAC, SEED + 100 * fold_idx
        )
        train_set, dev_set = set(train_ids), set(dev_ids)

        initial_models, continuation_models = fit_model_bundle(
            initial_records, continuation_records, train_set
        )

        dev_rows = [r for r in raw if int(r["question_id"]) in dev_set]
        test_rows = [r for r in raw if int(r["question_id"]) in test_set]

        utility_choice = choose_utility_policy(
            dev_rows, initial_models, continuation_models
        )
        selector_choice = choose_risk_policy(
            dev_rows,
            initial_models,
            continuation_models,
            adaptive_depth=False,
        )
        rair_choice = choose_risk_policy(
            dev_rows,
            initial_models,
            continuation_models,
            adaptive_depth=True,
        )

        up: UtilityParams = utility_choice["params"]
        sp: PolicyParams = selector_choice["params"]
        rp: PolicyParams = rair_choice["params"]

        fold_rows.append(
            {
                "fold": fold_idx,
                "train_questions": len(train_ids),
                "dev_questions": len(dev_ids),
                "test_questions": len(test_ids),
                "utility_cost_weight": up.cost_weight,
                "utility_score_threshold": up.score_threshold,
                "utility_dev_accuracy": utility_choice["accuracy"],
                "utility_dev_tokens": utility_choice["mean_tokens"],
                "selector_harm_weight": sp.harm_weight,
                "selector_cost_weight": sp.cost_weight,
                "selector_score_threshold": sp.score_threshold,
                "selector_harm_cap": sp.harm_cap,
                "selector_dev_accuracy": selector_choice["accuracy"],
                "selector_dev_tokens": selector_choice["mean_tokens"],
                "rair_harm_weight": rp.harm_weight,
                "rair_cost_weight": rp.cost_weight,
                "rair_score_threshold": rp.score_threshold,
                "rair_harm_cap": rp.harm_cap,
                "rair_dev_accuracy": rair_choice["accuracy"],
                "rair_dev_tokens": rair_choice["mean_tokens"],
                "rair_dev_harm": rair_choice["harmful_exposure_rate"],
                "selection_rule": rair_choice["selection_rule"],
            }
        )

        # Out-of-sample classifier diagnostics.
        for stage, records_by_protocol, models_by_protocol in (
            ("initial", initial_records, initial_models),
            ("continuation", continuation_records, continuation_models),
        ):
            for protocol in PROTOCOLS:
                test_records = filter_records(records_by_protocol[protocol], test_set)
                model_pair = models_by_protocol[protocol]
                for label, model in (
                    ("help", model_pair.help_model),
                    ("harm", model_pair.harm_model),
                ):
                    classifier_rows.append(
                        {
                            "fold": fold_idx,
                            "stage": stage,
                            "protocol": protocol,
                            "label": label,
                            **classifier_metrics(model, test_records, label),
                        }
                    )

                    names = (
                        INITIAL_FEATURES
                        if stage == "initial"
                        else feature_names_for_protocol(protocol)
                    )
                    coefs = extract_coefficients(model, names)
                    if coefs is not None:
                        for name, value in zip(names, coefs):
                            coefficient_rows.append(
                                {
                                    "fold": fold_idx,
                                    "stage": stage,
                                    "protocol": protocol,
                                    "label": label,
                                    "feature": name,
                                    "standardized_coefficient": value,
                                }
                            )

        for row in test_rows:
            methods = {
                "single": fixed_single(row),
                "fixed_self_r1": fixed_protocol(row, "self_refine", 1),
                "fixed_self_r2": fixed_protocol(row, "self_refine", 2),
                "fixed_critic_r1": fixed_protocol(row, "student_critic", 1),
                "fixed_critic_r2": fixed_protocol(row, "student_critic", 2),
                "fixed_debate_r0": fixed_protocol(row, "debate", 0),
                "fixed_debate_r1": fixed_protocol(row, "debate", 1),
                "fixed_debate_r2": fixed_protocol(row, "debate", 2),
                "utility_only_router": run_router(
                    row,
                    initial_models,
                    continuation_models,
                    up,
                    utility_only=True,
                    adaptive_depth=True,
                ),
                "risk_aware_selector": run_router(
                    row,
                    initial_models,
                    continuation_models,
                    sp,
                    utility_only=False,
                    adaptive_depth=False,
                ),
                "rair": run_router(
                    row,
                    initial_models,
                    continuation_models,
                    rp,
                    utility_only=False,
                    adaptive_depth=True,
                ),
            }
            predictions.append(
                {
                    "outer_fold": fold_idx,
                    "question_id": int(row["question_id"]),
                    "repetition": int(row["repetition"]),
                    "level": int(row["level"]),
                    "methods": methods,
                }
            )

    return predictions, fold_rows, classifier_rows, coefficient_rows


# =============================================================================
# Summaries and statistics
# =============================================================================


def summarize_methods(predictions: Sequence[dict]) -> List[dict]:
    rows = []
    for method in METHODS:
        vals = [p["methods"][method] for p in predictions]
        protocol_counts = Counter(v["protocol"] for v in vals)
        rows.append(
            {
                "method": method,
                "accuracy": mean(float(v["correct"]) for v in vals),
                "mean_tokens_per_question": mean(v["tokens"] for v in vals),
                "mean_round": mean(v["round"] for v in vals),
                "harmful_transition_exposures_per_question": mean(
                    v["harmful_exposures"] for v in vals
                ),
                "helpful_transition_exposures_per_question": mean(
                    v["helpful_exposures"] for v in vals
                ),
                "single_fraction": protocol_counts["single"] / len(vals),
                "self_refine_fraction": protocol_counts["self_refine"] / len(vals),
                "student_critic_fraction": protocol_counts["student_critic"] / len(vals),
                "debate_fraction": protocol_counts["debate"] / len(vals),
            }
        )
    return rows


def percentile(values: Sequence[float], q: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def clustered_bootstrap(predictions: Sequence[dict], n_bootstrap: int = BOOTSTRAP_SAMPLES) -> List[dict]:
    by_question = defaultdict(list)
    for row in predictions:
        by_question[row["question_id"]].append(row)
    qids = sorted(by_question)
    rng = random.Random(BOOTSTRAP_SEED)

    baselines = [m for m in METHODS if m != "rair"]
    dist = {b: {"acc": [], "tokens": [], "harm": []} for b in baselines}
    point = {r["method"]: r for r in summarize_methods(predictions)}

    for _ in range(n_bootstrap):
        sampled = []
        for qid in [rng.choice(qids) for _ in qids]:
            sampled.extend(by_question[qid])

        rair_vals = [x["methods"]["rair"] for x in sampled]
        ra = mean(float(x["correct"]) for x in rair_vals)
        rt = mean(x["tokens"] for x in rair_vals)
        rh = mean(x["harmful_exposures"] for x in rair_vals)

        for baseline in baselines:
            vals = [x["methods"][baseline] for x in sampled]
            ba = mean(float(x["correct"]) for x in vals)
            bt = mean(x["tokens"] for x in vals)
            bh = mean(x["harmful_exposures"] for x in vals)
            dist[baseline]["acc"].append(ra - ba)
            dist[baseline]["tokens"].append((bt - rt) / bt if bt > 0 else 0.0)
            dist[baseline]["harm"].append(rh - bh)

    out = []
    for baseline in baselines:
        out.append(
            {
                "comparison": f"rair - {baseline}",
                "accuracy_difference": point["rair"]["accuracy"] - point[baseline]["accuracy"],
                "accuracy_diff_ci95_low": percentile(dist[baseline]["acc"], 0.025),
                "accuracy_diff_ci95_high": percentile(dist[baseline]["acc"], 0.975),
                "token_savings_fraction": (
                    point[baseline]["mean_tokens_per_question"]
                    - point["rair"]["mean_tokens_per_question"]
                )
                / point[baseline]["mean_tokens_per_question"],
                "token_savings_ci95_low": percentile(dist[baseline]["tokens"], 0.025),
                "token_savings_ci95_high": percentile(dist[baseline]["tokens"], 0.975),
                "harmful_exposure_difference": (
                    point["rair"]["harmful_transition_exposures_per_question"]
                    - point[baseline]["harmful_transition_exposures_per_question"]
                ),
                "harm_diff_ci95_low": percentile(dist[baseline]["harm"], 0.025),
                "harm_diff_ci95_high": percentile(dist[baseline]["harm"], 0.975),
            }
        )
    return out


def routing_summary(predictions: Sequence[dict]) -> List[dict]:
    rows = []
    for method in ("utility_only_router", "risk_aware_selector", "rair"):
        vals = [p["methods"][method] for p in predictions]
        counts = Counter((v["protocol"], v["round"]) for v in vals)
        for (protocol, round_id), n in sorted(counts.items()):
            subset = [v for v in vals if v["protocol"] == protocol and v["round"] == round_id]
            rows.append(
                {
                    "method": method,
                    "protocol": protocol,
                    "round": round_id,
                    "count": n,
                    "fraction": n / len(vals),
                    "accuracy": mean(float(v["correct"]) for v in subset),
                    "mean_tokens": mean(v["tokens"] for v in subset),
                }
            )
    return rows


def summarize_classifier_metrics(rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["protocol"], row["label"])].append(row)
    out = []
    for (stage, protocol, label), vals in sorted(grouped.items()):
        aucs = [v["roc_auc"] for v in vals if v["roc_auc"] is not None]
        aps = [v["average_precision"] for v in vals if v["average_precision"] is not None]
        prev = [v["prevalence"] for v in vals if v["prevalence"] is not None]
        out.append(
            {
                "stage": stage,
                "protocol": protocol,
                "label": label,
                "mean_roc_auc": mean(aucs) if aucs else None,
                "mean_average_precision": mean(aps) if aps else None,
                "mean_prevalence": mean(prev) if prev else None,
            }
        )
    return out


# =============================================================================
# Plots and quick read
# =============================================================================


def make_plots(out_dir: Path, method_summary: Sequence[dict], routing: Sequence[dict]) -> None:
    md = {r["method"]: r for r in method_summary}

    plt.figure(figsize=(9, 5.5))
    for method in METHODS:
        x = md[method]["mean_tokens_per_question"]
        y = 100 * md[method]["accuracy"]
        plt.scatter(x, y, s=65)
        plt.annotate(method, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    plt.xlabel("Mean tokens per question")
    plt.ylabel("Accuracy (%)")
    plt.title("Experiment H -- Accuracy vs. Token Cost")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_tokens.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for method in METHODS:
        x = md[method]["mean_tokens_per_question"]
        y = 100 * md[method]["harmful_transition_exposures_per_question"]
        plt.scatter(x, y, s=65)
        plt.annotate(method, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    plt.xlabel("Mean tokens per question")
    plt.ylabel("Harmful transitions taken / question (%)")
    plt.title("Experiment H -- Interaction Risk vs. Token Cost")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "risk_vs_tokens.png", dpi=300)
    plt.close()

    rair_rows = [r for r in routing if r["method"] == "rair"]
    labels = [f"{r['protocol']} R{r['round']}" for r in rair_rows]
    vals = [100 * r["fraction"] for r in rair_rows]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, vals)
    plt.ylabel("Questions routed (%)")
    plt.title("Experiment H -- RAIR Routing Distribution")
    plt.xticks(rotation=30, ha="right")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "rair_routing_distribution.png", dpi=300)
    plt.close()


def write_quick_read(
    path: Path,
    method_summary: Sequence[dict],
    comparisons: Sequence[dict],
    classifier_summary: Sequence[dict],
) -> None:
    lines = [
        "EXPERIMENT H -- RISK-AWARE INTERACTION ROUTER (RAIR)",
        "=" * 78,
        "",
        "METHOD SUMMARY",
    ]
    for x in method_summary:
        lines.append(
            f"  {x['method']:<23} "
            f"acc={100*x['accuracy']:.2f}% | "
            f"tokens/q={x['mean_tokens_per_question']:.1f} | "
            f"harm={100*x['harmful_transition_exposures_per_question']:.2f}% | "
            f"self={100*x['self_refine_fraction']:.1f}% "
            f"critic={100*x['student_critic_fraction']:.1f}% "
            f"debate={100*x['debate_fraction']:.1f}% "
            f"stop={100*x['single_fraction']:.1f}%"
        )

    lines += ["", "RAIR PAIRED COMPARISONS"]
    for x in comparisons:
        lines.append(
            f"  {x['comparison']}: "
            f"accuracy={100*x['accuracy_difference']:+.2f} pp "
            f"[{100*x['accuracy_diff_ci95_low']:+.2f}, {100*x['accuracy_diff_ci95_high']:+.2f}] | "
            f"token savings={100*x['token_savings_fraction']:+.2f}% | "
            f"harm diff={100*x['harmful_exposure_difference']:+.2f} pp"
        )

    lines += ["", "OUT-OF-SAMPLE HELP/HARM CLASSIFIERS"]
    for x in classifier_summary:
        auc = "NA" if x["mean_roc_auc"] is None else f"{x['mean_roc_auc']:.3f}"
        ap = "NA" if x["mean_average_precision"] is None else f"{x['mean_average_precision']:.3f}"
        lines.append(
            f"  {x['stage']:<12} {x['protocol']:<15} {x['label']:<4} "
            f"AUC={auc} | AP={ap} | prevalence={x['mean_prevalence']:.3f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Validation
# =============================================================================


def validate_raw(raw: Sequence[dict]) -> None:
    if not raw:
        raise ValueError("Input raw_results.jsonl is empty")
    required_top = {"question_id", "repetition", "level", "single", *PROTOCOLS}
    for i, row in enumerate(raw[:10]):
        missing = required_top - set(row)
        if missing:
            raise ValueError(f"Row {i} missing required fields: {sorted(missing)}")
        for protocol in PROTOCOLS:
            needed = {"0", "1", "2"}
            missing_rounds = needed - set(row[protocol])
            if missing_rounds:
                raise ValueError(
                    f"Row {i} protocol {protocol} missing rounds {sorted(missing_rounds)}"
                )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-raw", type=Path, default=B_RAW)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_jsonl(args.b_raw)
    validate_raw(raw)

    print("=" * 80)
    print("EXPERIMENT H -- RISK-AWARE INTERACTION ROUTER (RAIR)")
    print("=" * 80)
    print(f"Input rows: {len(raw)}")
    print(f"Unique questions: {len({int(r['question_id']) for r in raw})}")
    print(f"Protocols: {', '.join(PROTOCOLS)}")
    print("Runtime features use gold: False")
    print("Runtime features use MATH difficulty: False")
    print("New LLM inference: False")
    print("=" * 80)

    predictions, fold_rows, classifier_rows, coefficient_rows = crossfit(raw)
    method_summary = summarize_methods(predictions)
    comparisons = clustered_bootstrap(predictions, args.bootstrap_samples)
    routing = routing_summary(predictions)
    classifier_summary = summarize_classifier_metrics(classifier_rows)

    write_jsonl(args.output_dir / "crossfit_predictions.jsonl", predictions)
    write_csv(args.output_dir / "method_summary.csv", method_summary)
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    write_csv(args.output_dir / "fold_parameters.csv", fold_rows)
    write_csv(args.output_dir / "classifier_metrics_by_fold.csv", classifier_rows)
    write_csv(args.output_dir / "classifier_summary.csv", classifier_summary)
    write_csv(args.output_dir / "feature_coefficients.csv", coefficient_rows)
    write_csv(args.output_dir / "routing_summary.csv", routing)

    final = {
        "experiment": "H - Risk-Aware Interaction Router",
        "method_name": "RAIR",
        "research_question": (
            "Can a learned controller select whether and how an LLM should interact, "
            "using predicted utility, harm, and cost to improve the joint "
            "accuracy-cost-risk trade-off?"
        ),
        "design": {
            "source": "Experiment B / Experiment 7 traces",
            "new_llm_inference": False,
            "outer_crossfit_folds": N_FOLDS,
            "split_unit": "question_id",
            "runtime_features_use_gold": False,
            "difficulty_used_as_runtime_feature": False,
            "difficulty_used_only_for_split_stratification": True,
            "protocols": list(PROTOCOLS),
            "initial_actions": ["STOP", "SELF_REFINE", "STUDENT_CRITIC", "DEBATE"],
            "adaptive_depth": True,
            "max_protocol_round": MAX_PROTOCOL_ROUND,
            "accuracy_tolerance_for_dev_selection": ACCURACY_TOLERANCE,
            "important_constraint": (
                "Protocol is selected at the common initial state and then held fixed; "
                "only depth is adapted within that protocol because Experiment B does "
                "not contain counterfactual mid-trajectory protocol-switch traces."
            ),
            "debate_cost_note": (
                "The router observes Single before choosing Debate. Because Experiment B "
                "generated Debate as a separate path, reported RAIR Debate cost conservatively "
                "adds Single tokens to Debate protocol tokens."
            ),
        },
        "method_summary": method_summary,
        "paired_comparisons": comparisons,
        "fold_parameters": fold_rows,
        "classifier_summary": classifier_summary,
        "routing_summary": routing,
    }
    save_json(args.output_dir / "final_summary.json", final)

    make_plots(args.output_dir, method_summary, routing)
    write_quick_read(
        args.output_dir / "quick_read.txt",
        method_summary,
        comparisons,
        classifier_summary,
    )

    print((args.output_dir / "quick_read.txt").read_text(encoding="utf-8"))
    print("Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
