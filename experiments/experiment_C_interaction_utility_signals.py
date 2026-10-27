#!/usr/bin/env python3
"""
Experiment C -- Interaction Utility Signals

Research question:
    Which observable signals predict whether another interaction/update is likely
    to help, hurt, or be unnecessary?

This experiment intentionally reuses the per-question traces produced by
Experiment B (Experiment 7). It performs no new LLM inference.

Signals studied from the existing traces:
    1. Debate disagreement before the next interaction round.
    2. Student--Critic VERDICT=REJECT before the student update.
    3. Self-Refine answer instability before an additional refinement round.
    4. MATH difficulty as an explanatory variable only (not a deployment gate).

Main outcome for every transition:
    delta = correctness_after - correctness_before in {-1, 0, +1}

We report:
    - P(wrong -> correct | signal)
    - P(correct -> wrong | signal)
    - mean accuracy delta
    - before/after accuracy
    - signal coverage
    - clustered bootstrap 95% CI over question IDs

The purpose is to identify which signals are worth testing as routing/stopping
rules in Experiment D. This script does NOT implement adaptive routing yet.
"""

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

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


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", "/home/isa1909/projects/def-sigmagao/isa1909")
)
EXP_B_DIR = Path(
    os.environ.get(
        "EXP_B_DIR",
        str(PROJECT_ROOT / "results" / "experiment_7_round_scaling_difficulty"),
    )
)
RAW_RESULTS = Path(
    os.environ.get("RAW_RESULTS", str(EXP_B_DIR / "raw_results.jsonl"))
)
RESULTS_DIR = Path(
    os.environ.get(
        "RESULTS_DIR",
        str(PROJECT_ROOT / "results" / "experiment_C_interaction_utility_signals"),
    )
)

BOOTSTRAP_SAMPLES = int(os.environ.get("BOOTSTRAP_SAMPLES", "2000"))
BOOTSTRAP_SEED = int(os.environ.get("BOOTSTRAP_SEED", "20261005"))


# ============================================================
# I/O AND MATH HELPERS
# ============================================================


def save_json(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def parse_math(text):
    if text is None:
        return []
    value = str(text).strip()
    if not value:
        return []
    if not ((value.startswith("$") and value.endswith("$")) or "\\boxed" in value):
        value = f"${value}$"
    try:
        return math_parse(
            value,
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        )
    except Exception:
        return []


def semantic_match(answer_a, answer_b):
    if answer_a is None or answer_b is None:
        return False
    try:
        parsed_a = parse_math(answer_a)
        parsed_b = parse_math(answer_b)
        return bool(parsed_a and parsed_b and math_verify(parsed_a, parsed_b))
    except Exception:
        return False


def get_answer(output_obj):
    if not output_obj:
        return None
    if isinstance(output_obj, dict):
        return output_obj.get("answer")
    return None


# ============================================================
# TRANSITION RECORDS
# ============================================================


def make_transition(
    row,
    *,
    method,
    transition,
    signal_family,
    signal_name,
    signal_value,
    correct_before,
    correct_after,
):
    before = bool(correct_before)
    after = bool(correct_after)
    return {
        "repetition": int(row["repetition"]),
        "question_id": int(row["question_id"]),
        "dataset_index": int(row["dataset_index"]),
        "level": row.get("level"),
        "type": row.get("type"),
        "method": method,
        "transition": transition,
        "signal_family": signal_family,
        "signal_name": signal_name,
        "signal_value": bool(signal_value),
        "correct_before": before,
        "correct_after": after,
        "delta": int(after) - int(before),
        "helped": (not before) and after,
        "harmed": before and (not after),
    }


def build_signal_records(raw_rows):
    records = []

    for row in raw_rows:
        self_refine = row["self_refine"]
        student_critic = row["student_critic"]
        debate = row["debate"]

        # ----------------------------------------------------
        # Signal 1: Debate disagreement before another round.
        # R=r disagreement predicts utility of R=r -> R=r+1.
        # ----------------------------------------------------
        for r in (0, 1, 2):
            before = debate[str(r)]
            after = debate[str(r + 1)]
            records.append(
                make_transition(
                    row,
                    method="debate",
                    transition=f"R{r}->R{r+1}",
                    signal_family="peer_disagreement",
                    signal_name="disagreement",
                    signal_value=before["disagreement"],
                    correct_before=before["correct"],
                    correct_after=after["correct"],
                )
            )

        # ----------------------------------------------------
        # Signal 2: Critic verdict before the student update.
        # Critic verdict at round r predicts utility S(r-1)->S(r).
        # ----------------------------------------------------
        for r in (1, 2, 3):
            before = student_critic[str(r - 1)]
            after = student_critic[str(r)]
            verdict = str(after.get("verdict") or "UNKNOWN").upper()
            records.append(
                make_transition(
                    row,
                    method="student_critic",
                    transition=f"R{r-1}->R{r}",
                    signal_family="critic_verdict",
                    signal_name="critic_reject",
                    signal_value=(verdict == "REJECT"),
                    correct_before=before["correct"],
                    correct_after=after["correct"],
                )
            )

        # ----------------------------------------------------
        # Signal 3: Self-Refine answer instability.
        # After observing R=r, test whether a changed answer predicts
        # utility from one MORE round: R=r -> R=r+1.
        # Only r=1,2 have a previous answer and a next round.
        # ----------------------------------------------------
        for r in (1, 2):
            previous = self_refine[str(r - 1)]
            current = self_refine[str(r)]
            nxt = self_refine[str(r + 1)]

            prev_answer = get_answer(previous.get("output"))
            current_answer = get_answer(current.get("output"))
            stable = semantic_match(prev_answer, current_answer)

            records.append(
                make_transition(
                    row,
                    method="self_refine",
                    transition=f"R{r}->R{r+1}",
                    signal_family="answer_stability",
                    signal_name="answer_unstable",
                    signal_value=(not stable),
                    correct_before=current["correct"],
                    correct_after=nxt["correct"],
                )
            )

    return records


# ============================================================
# STATISTICS
# ============================================================


def safe_div(num, den):
    return num / den if den else None


def summarize_records(records):
    n = len(records)
    before_correct = sum(r["correct_before"] for r in records)
    after_correct = sum(r["correct_after"] for r in records)
    wrong_before = n - before_correct
    helped = sum(r["helped"] for r in records)
    harmed = sum(r["harmed"] for r in records)

    return {
        "n_records": n,
        "n_unique_questions": len({r["question_id"] for r in records}),
        "before_accuracy": safe_div(before_correct, n),
        "after_accuracy": safe_div(after_correct, n),
        "mean_delta_accuracy": safe_div(after_correct - before_correct, n),
        "help_count": helped,
        "harm_count": harmed,
        "help_rate_among_wrong": safe_div(helped, wrong_before),
        "harm_rate_among_correct": safe_div(harmed, before_correct),
        "net_transition_rate": safe_div(helped - harmed, n),
    }


def clustered_bootstrap_mean_delta(records, n_bootstrap, seed):
    """Bootstrap question IDs so repeated stochastic runs stay in the same cluster."""
    by_question = defaultdict(list)
    for record in records:
        by_question[record["question_id"]].append(record)

    question_ids = sorted(by_question)
    if not question_ids:
        return None, None

    rng = random.Random(seed)
    values = []
    for _ in range(n_bootstrap):
        sampled_ids = [rng.choice(question_ids) for _ in question_ids]
        deltas = []
        for qid in sampled_ids:
            deltas.extend(r["delta"] for r in by_question[qid])
        values.append(mean(deltas))

    values.sort()
    lo_idx = int(0.025 * (len(values) - 1))
    hi_idx = int(0.975 * (len(values) - 1))
    return values[lo_idx], values[hi_idx]


def add_ci(summary, records, seed_offset=0):
    lo, hi = clustered_bootstrap_mean_delta(
        records,
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_SEED + seed_offset,
    )
    summary["mean_delta_ci95_low"] = lo
    summary["mean_delta_ci95_high"] = hi
    return summary


# ============================================================
# SIGNAL ANALYSIS
# ============================================================


def summarize_signal_groups(records):
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["signal_family"],
            record["method"],
            record["transition"],
            record["signal_name"],
            record["signal_value"],
        )
        grouped[key].append(record)

    rows = []
    for idx, (key, group) in enumerate(sorted(grouped.items(), key=str)):
        signal_family, method, transition, signal_name, signal_value = key
        summary = add_ci(summarize_records(group), group, idx)
        rows.append(
            {
                "signal_family": signal_family,
                "method": method,
                "transition": transition,
                "signal_name": signal_name,
                "signal_value": signal_value,
                **summary,
            }
        )
    return rows


def summarize_candidate_routing_signals(records):
    """
    Collapse across eligible transitions to estimate whether a TRUE signal means
    'another interaction/update is more useful than when the signal is FALSE'.
    """
    grouped = defaultdict(list)
    for record in records:
        key = (record["signal_family"], record["method"], record["signal_name"])
        grouped[key].append(record)

    rows = []
    for idx, (key, group) in enumerate(sorted(grouped.items(), key=str)):
        family, method, signal_name = key
        true_group = [r for r in group if r["signal_value"]]
        false_group = [r for r in group if not r["signal_value"]]

        true_summary = summarize_records(true_group) if true_group else None
        false_summary = summarize_records(false_group) if false_group else None

        coverage = safe_div(len(true_group), len(group))
        utility_true = true_summary["mean_delta_accuracy"] if true_summary else None
        utility_false = false_summary["mean_delta_accuracy"] if false_summary else None
        utility_lift = (
            utility_true - utility_false
            if utility_true is not None and utility_false is not None
            else None
        )

        # Bootstrap the difference in utility between signal TRUE and FALSE.
        # Cluster by question ID and preserve all repetitions/transitions per sampled ID.
        question_ids = sorted({r["question_id"] for r in group})
        by_question = defaultdict(list)
        for r in group:
            by_question[r["question_id"]].append(r)

        rng = random.Random(BOOTSTRAP_SEED + 10000 + idx)
        boot_lifts = []
        if question_ids and true_group and false_group:
            for _ in range(BOOTSTRAP_SAMPLES):
                sampled_ids = [rng.choice(question_ids) for _ in question_ids]
                sampled = []
                for qid in sampled_ids:
                    sampled.extend(by_question[qid])
                t = [r["delta"] for r in sampled if r["signal_value"]]
                f = [r["delta"] for r in sampled if not r["signal_value"]]
                if t and f:
                    boot_lifts.append(mean(t) - mean(f))

        if boot_lifts:
            boot_lifts.sort()
            lo = boot_lifts[int(0.025 * (len(boot_lifts) - 1))]
            hi = boot_lifts[int(0.975 * (len(boot_lifts) - 1))]
        else:
            lo = hi = None

        rows.append(
            {
                "signal_family": family,
                "method": method,
                "signal_name": signal_name,
                "signal_true_means": "candidate_for_more_interaction",
                "coverage_signal_true": coverage,
                "n_records": len(group),
                "n_unique_questions": len(question_ids),
                "utility_when_true": utility_true,
                "utility_when_false": utility_false,
                "utility_lift_true_minus_false": utility_lift,
                "utility_lift_ci95_low": lo,
                "utility_lift_ci95_high": hi,
                "help_rate_when_true": (
                    true_summary["help_rate_among_wrong"] if true_summary else None
                ),
                "harm_rate_when_true": (
                    true_summary["harm_rate_among_correct"] if true_summary else None
                ),
                "help_rate_when_false": (
                    false_summary["help_rate_among_wrong"] if false_summary else None
                ),
                "harm_rate_when_false": (
                    false_summary["harm_rate_among_correct"] if false_summary else None
                ),
            }
        )

    return rows


# ============================================================
# DIFFICULTY ANALYSIS (EXPLANATORY ONLY)
# ============================================================


def build_difficulty_records(raw_rows):
    """First-round utility by MATH level. Difficulty is not treated as a routing gate."""
    records = []
    for row in raw_rows:
        level = row.get("level")
        if level is None:
            continue

        # Self-Refine R0 -> R1
        before = row["self_refine"]["0"]["correct"]
        after = row["self_refine"]["1"]["correct"]
        records.append(
            {
                "repetition": row["repetition"],
                "question_id": row["question_id"],
                "level": level,
                "method": "self_refine",
                "transition": "R0->R1",
                "correct_before": before,
                "correct_after": after,
                "delta": int(after) - int(before),
                "helped": (not before) and after,
                "harmed": before and (not after),
            }
        )

        # Student-Critic R0 -> R1
        before = row["student_critic"]["0"]["correct"]
        after = row["student_critic"]["1"]["correct"]
        records.append(
            {
                "repetition": row["repetition"],
                "question_id": row["question_id"],
                "level": level,
                "method": "student_critic",
                "transition": "R0->R1",
                "correct_before": before,
                "correct_after": after,
                "delta": int(after) - int(before),
                "helped": (not before) and after,
                "harmed": before and (not after),
            }
        )

        # Debate R0 -> R1
        before = row["debate"]["0"]["correct"]
        after = row["debate"]["1"]["correct"]
        records.append(
            {
                "repetition": row["repetition"],
                "question_id": row["question_id"],
                "level": level,
                "method": "debate",
                "transition": "R0->R1",
                "correct_before": before,
                "correct_after": after,
                "delta": int(after) - int(before),
                "helped": (not before) and after,
                "harmed": before and (not after),
            }
        )

    return records


def summarize_difficulty(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["method"], r["level"])].append(r)

    rows = []
    for idx, ((method, level), group) in enumerate(sorted(grouped.items())):
        summary = add_ci(summarize_records(group), group, 20000 + idx)
        rows.append(
            {
                "method": method,
                "level": level,
                "transition": "R0->R1",
                **summary,
            }
        )
    return rows


# ============================================================
# COMPACT HUMAN-READABLE SUMMARY
# ============================================================


def pct(value):
    return "NA" if value is None else f"{100 * value:.2f}%"


def write_text_summary(path, candidate_rows, difficulty_rows, n_raw_rows):
    lines = []
    lines.append("EXPERIMENT C -- INTERACTION UTILITY SIGNALS")
    lines.append("=" * 60)
    lines.append(f"Input trace rows: {n_raw_rows}")
    lines.append(
        "Goal: identify observable signals that predict whether another interaction/update helps."
    )
    lines.append("")
    lines.append("Candidate routing signals")
    lines.append("-" * 60)

    ranked = sorted(
        candidate_rows,
        key=lambda r: (
            r["utility_lift_true_minus_false"] is None,
            -(r["utility_lift_true_minus_false"] or -999),
        ),
    )

    for row in ranked:
        lines.append(
            f"{row['method']} | {row['signal_name']}: "
            f"coverage={pct(row['coverage_signal_true'])}, "
            f"utility(TRUE)={pct(row['utility_when_true'])}, "
            f"utility(FALSE)={pct(row['utility_when_false'])}, "
            f"lift={pct(row['utility_lift_true_minus_false'])}, "
            f"95% CI=[{pct(row['utility_lift_ci95_low'])}, {pct(row['utility_lift_ci95_high'])}]"
        )
        lines.append(
            f"    TRUE: help={pct(row['help_rate_when_true'])}, "
            f"harm={pct(row['harm_rate_when_true'])}"
        )

    lines.append("")
    lines.append("Difficulty (descriptive only; not a deployment routing signal)")
    lines.append("-" * 60)
    for row in difficulty_rows:
        lines.append(
            f"{row['method']} | Level {row['level']}: "
            f"delta={pct(row['mean_delta_accuracy'])}, "
            f"help={pct(row['help_rate_among_wrong'])}, "
            f"harm={pct(row['harm_rate_among_correct'])}"
        )

    lines.append("")
    lines.append("Interpretation rule for Experiment D")
    lines.append("-" * 60)
    lines.append(
        "Prefer signals with positive utility when TRUE, lower utility when FALSE, "
        "and a positive utility lift whose clustered-bootstrap CI is reasonably stable."
    )
    lines.append(
        "Do not select a routing policy here; Experiment C only identifies candidate signals."
    )
    lines.append(
        "Critic confidence is not analyzed because Experiment B did not record a calibrated confidence score."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# MAIN
# ============================================================


def main():
    if not RAW_RESULTS.exists():
        raise SystemExit(
            f"Experiment B raw trace file not found:\n  {RAW_RESULTS}\n\n"
            "Set RAW_RESULTS=/path/to/raw_results.jsonl or EXP_B_DIR=/path/to/experiment_7_results."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT C - INTERACTION UTILITY SIGNALS")
    print("=" * 80)
    print("Input:", RAW_RESULTS)
    print("Output:", RESULTS_DIR)
    print("Bootstrap samples:", BOOTSTRAP_SAMPLES)
    print("No new model inference will be run.")
    print("=" * 80)

    raw_rows = load_jsonl(RAW_RESULTS)
    if not raw_rows:
        raise SystemExit("raw_results.jsonl is empty")

    repetitions = sorted({int(r["repetition"]) for r in raw_rows})
    question_ids = sorted({int(r["question_id"]) for r in raw_rows})

    signal_records = build_signal_records(raw_rows)
    signal_group_rows = summarize_signal_groups(signal_records)
    candidate_rows = summarize_candidate_routing_signals(signal_records)

    difficulty_records = build_difficulty_records(raw_rows)
    difficulty_rows = summarize_difficulty(difficulty_records)

    write_csv(RESULTS_DIR / "signal_group_summary.csv", signal_group_rows)
    write_csv(RESULTS_DIR / "routing_signal_comparison.csv", candidate_rows)
    write_csv(RESULTS_DIR / "difficulty_first_round_utility.csv", difficulty_rows)

    final_summary = {
        "experiment": "C - Interaction Utility Signals",
        "research_question": (
            "Which observable signals predict whether another interaction round or update "
            "is likely to improve rather than degrade the answer?"
        ),
        "source_experiment": "B / Experiment 7",
        "new_llm_inference": False,
        "input_file": str(RAW_RESULTS),
        "trace_rows": len(raw_rows),
        "repetitions": repetitions,
        "unique_questions": len(question_ids),
        "signals": [
            "debate disagreement",
            "critic REJECT verdict",
            "self-refine answer instability",
        ],
        "difficulty_note": (
            "MATH level is analyzed only as an explanatory variable because it is benchmark metadata, "
            "not a generally available runtime signal."
        ),
        "critic_confidence_note": (
            "Experiment B did not record a calibrated critic confidence score, so confidence is not "
            "analyzed in Experiment C. The structured ACCEPT/REJECT verdict is analyzed instead."
        ),
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "question_id cluster",
            "reason": "the same questions appear across stochastic repetitions",
        },
        "candidate_signal_comparison": candidate_rows,
        "difficulty_first_round_utility": difficulty_rows,
    }
    save_json(RESULTS_DIR / "final_summary.json", final_summary)

    write_text_summary(
        RESULTS_DIR / "quick_read.txt",
        candidate_rows,
        difficulty_rows,
        len(raw_rows),
    )

    print(f"Loaded {len(raw_rows)} trace rows from {len(repetitions)} repetitions.")
    print(f"Unique questions: {len(question_ids)}")
    print("\nCandidate signal summary:")
    for row in candidate_rows:
        print(
            f"  {row['method']:<16} {row['signal_name']:<18} "
            f"coverage={pct(row['coverage_signal_true']):>8} "
            f"utility_true={pct(row['utility_when_true']):>8} "
            f"utility_false={pct(row['utility_when_false']):>8} "
            f"lift={pct(row['utility_lift_true_minus_false']):>8}"
        )

    print("\nSaved:")
    print(" ", RESULTS_DIR / "final_summary.json")
    print(" ", RESULTS_DIR / "routing_signal_comparison.csv")
    print(" ", RESULTS_DIR / "signal_group_summary.csv")
    print(" ", RESULTS_DIR / "difficulty_first_round_utility.csv")
    print(" ", RESULTS_DIR / "quick_read.txt")


if __name__ == "__main__":
    main()
