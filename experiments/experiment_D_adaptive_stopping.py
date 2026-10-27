#!/usr/bin/env python3
"""
Experiment D - Adaptive Self-Refine

Research question:
    Can adaptive stopping preserve most of the accuracy gain of fixed-round
    Self-Refine while reducing unnecessary interaction cost?

Design:
    - Dataset: MATH (jeggers/competition_math), test split
    - 500 questions, balanced across Levels 1-5 (100 per level)
    - 3 repetitions
    - Same Qwen2.5-7B-Instruct base model
    - Fixed baselines: R0, R1, R2, R3
    - Adaptive policy:
          Always perform R1.
          After each revision, compare the current and previous final answers
          semantically (Math-Verify when possible).
          If equivalent -> STOP.
          Otherwise continue, up to R3.

Methodological note:
    The primary comparison is derived from one full R3 trajectory per question.
    Fixed R1/R2/R3 and Adaptive therefore use the exact same sampled trajectory.
    This avoids extra sampling noise. Adaptive token counts are counterfactual:
    only stages that the adaptive policy would have executed are counted.

    Optional --runtime-validation performs a second truly adaptive execution,
    actually skipping later generations after stopping, to measure wall time.

Outputs:
    results/experiment_D_adaptive_stopping/
        selected_questions.json
        raw_results.jsonl
        repetition_1_summary.json
        repetition_2_summary.json
        repetition_3_summary.json
        final_summary.json
        comparison_summary.csv
        quick_read.txt
        runtime_validation_rep_*.json   (if enabled)
"""

import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from datasets import load_dataset
from vllm import LLM, SamplingParams

try:
    from math_verify import LatexExtractionConfig, parse, verify
    MATH_VERIFY_AVAILABLE = True
except Exception:
    MATH_VERIFY_AVAILABLE = False
    LatexExtractionConfig = None
    parse = None
    verify = None


PROJECT_ROOT = Path("/lustre06/project/6087921/isa1909")
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct"
RESULT_DIR = PROJECT_ROOT / "results" / "experiment_D_adaptive_stopping"

DATASET_NAME = "jeggers/competition_math"
DATASET_SPLIT = "test"
QUESTIONS_PER_LEVEL = 100
LEVELS = [1, 2, 3, 4, 5]
N_REPETITIONS = 3
MAX_ROUNDS = 3

BASE_SAMPLE_SEED = 20261007
BASE_GENERATION_SEED = 20261100

TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS_ANSWER = 768
MAX_TOKENS_FEEDBACK = 512
MAX_TOKENS_REVISION = 768

SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. "
    "Give a correct, concise solution and put the final answer in \\boxed{...}."
)

SELF_FEEDBACK_PROMPT = """Review the solution below carefully.

Problem:
{problem}

Current solution:
{answer}

Identify any mathematical, logical, or calculation error. If the solution is
already correct, say so clearly. Focus only on issues that could change the
final answer."""

SELF_REVISION_PROMPT = """Solve the problem again using the feedback below.

Problem:
{problem}

Previous solution:
{answer}

Feedback:
{feedback}

Return a corrected concise solution. Put the final answer in \\boxed{{...}}.
If the previous solution is already correct, preserve it rather than changing
it unnecessarily."""


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def level_to_int(level: Any) -> int:
    m = re.search(r"([1-5])", str(level))
    if not m:
        raise ValueError(f"Could not parse MATH level: {level!r}")
    return int(m.group(1))


def extract_boxed(text: str) -> Optional[str]:
    if not text:
        return None
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:i].strip()
    return None


def normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = text.strip()
    boxed = extract_boxed(text)
    if boxed:
        text = boxed
    text = text.replace("$", "")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = re.sub(r"\s+", "", text)
    return text.rstrip(".")


def _mv_parse(text: str):
    if not MATH_VERIFY_AVAILABLE or not text:
        return None
    attempts = []
    if LatexExtractionConfig is not None:
        attempts.append(lambda: parse(text, extraction_config=[LatexExtractionConfig()]))
    attempts.append(lambda: parse(text))
    for fn in attempts:
        try:
            result = fn()
            if result:
                return result
        except Exception:
            continue
    return None


def semantically_equivalent(answer_a: str, answer_b: str) -> Tuple[bool, str]:
    """Routing signal. Never uses the gold answer."""
    parsed_a = _mv_parse(answer_a)
    parsed_b = _mv_parse(answer_b)
    if parsed_a is not None and parsed_b is not None:
        try:
            return bool(verify(parsed_a, parsed_b)), "math_verify"
        except Exception:
            pass
    norm_a = normalize_answer(answer_a)
    norm_b = normalize_answer(answer_b)
    return (norm_a != "" and norm_a == norm_b), "normalized_fallback"


def is_correct(model_answer: str, reference_solution: str) -> Tuple[bool, str]:
    parsed_pred = _mv_parse(model_answer)
    parsed_gold = _mv_parse(reference_solution)
    if parsed_pred is not None and parsed_gold is not None:
        try:
            return bool(verify(parsed_gold, parsed_pred)), "math_verify"
        except Exception:
            pass
    pred = normalize_answer(model_answer)
    gold = normalize_answer(reference_solution)
    return (pred != "" and pred == gold), "normalized_fallback"


def make_chat(tokenizer, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch(
    llm: LLM,
    prompts: List[str],
    max_tokens: int,
    seed: int,
) -> Tuple[List[str], List[Dict[str, int]], float]:
    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens,
        seed=seed,
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    wall_time = time.perf_counter() - start

    texts = []
    stats = []
    for out in outputs:
        item = out.outputs[0]
        prompt_tokens = len(out.prompt_token_ids or [])
        output_tokens = len(item.token_ids or [])
        texts.append(item.text)
        stats.append({
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        })
    return texts, stats, wall_time


def load_or_create_sample(result_dir: Path) -> List[Dict[str, Any]]:
    selected_path = result_dir / "selected_questions.json"
    if selected_path.exists():
        with selected_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    by_level: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(ds):
        level = level_to_int(row["level"])
        by_level[level].append({
            "question_id": idx,
            "problem": row["problem"],
            "solution": row["solution"],
            "level": level,
            "type": row.get("type"),
        })

    rng = random.Random(BASE_SAMPLE_SEED)
    selected = []
    for level in LEVELS:
        pool = by_level[level]
        if len(pool) < QUESTIONS_PER_LEVEL:
            raise RuntimeError(
                f"Level {level} has only {len(pool)} questions; need {QUESTIONS_PER_LEVEL}."
            )
        selected.extend(rng.sample(pool, QUESTIONS_PER_LEVEL))

    selected.sort(key=lambda x: (x["level"], x["question_id"]))
    dump_json(selected_path, selected)
    return selected


def run_full_trajectory(
    llm: LLM,
    tokenizer,
    questions: List[Dict[str, Any]],
    repetition: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    seed_base = BASE_GENERATION_SEED + 1000 * repetition
    n = len(questions)

    trajectories = [
        {
            **q,
            "repetition": repetition,
            "answers": {},
            "feedback": {},
            "tokens": {},
            "stage_wall_time": {},
        }
        for q in questions
    ]

    prompts = [
        make_chat(tokenizer, f"{q['problem']}\n\nSolve the problem.")
        for q in questions
    ]
    answers, token_stats, wall = generate_batch(
        llm, prompts, MAX_TOKENS_ANSWER, seed_base
    )
    wall_per_item = wall / max(n, 1)
    for i in range(n):
        trajectories[i]["answers"]["0"] = answers[i]
        trajectories[i]["tokens"]["answer_0"] = token_stats[i]
        trajectories[i]["stage_wall_time"]["answer_0"] = wall_per_item

    for r in range(1, MAX_ROUNDS + 1):
        feedback_prompts = [
            make_chat(
                tokenizer,
                SELF_FEEDBACK_PROMPT.format(
                    problem=q["problem"],
                    answer=trajectories[i]["answers"][str(r - 1)],
                ),
            )
            for i, q in enumerate(questions)
        ]
        feedback, fb_stats, fb_wall = generate_batch(
            llm,
            feedback_prompts,
            MAX_TOKENS_FEEDBACK,
            seed_base + 10 * r + 1,
        )
        fb_wall_per_item = fb_wall / max(n, 1)
        for i in range(n):
            trajectories[i]["feedback"][str(r)] = feedback[i]
            trajectories[i]["tokens"][f"feedback_{r}"] = fb_stats[i]
            trajectories[i]["stage_wall_time"][f"feedback_{r}"] = fb_wall_per_item

        revision_prompts = [
            make_chat(
                tokenizer,
                SELF_REVISION_PROMPT.format(
                    problem=q["problem"],
                    answer=trajectories[i]["answers"][str(r - 1)],
                    feedback=trajectories[i]["feedback"][str(r)],
                ),
            )
            for i, q in enumerate(questions)
        ]
        revisions, rev_stats, rev_wall = generate_batch(
            llm,
            revision_prompts,
            MAX_TOKENS_REVISION,
            seed_base + 10 * r + 2,
        )
        rev_wall_per_item = rev_wall / max(n, 1)
        for i in range(n):
            trajectories[i]["answers"][str(r)] = revisions[i]
            trajectories[i]["tokens"][f"answer_{r}"] = rev_stats[i]
            trajectories[i]["stage_wall_time"][f"answer_{r}"] = rev_wall_per_item

    reference_wall = sum(sum(t["stage_wall_time"].values()) for t in trajectories) / max(n, 1)
    return trajectories, {"reference_pipeline_wall_seconds": reference_wall}


def cumulative_tokens_to_round(traj: Dict[str, Any], round_idx: int) -> int:
    total = traj["tokens"]["answer_0"]["total_tokens"]
    for r in range(1, round_idx + 1):
        total += traj["tokens"][f"feedback_{r}"]["total_tokens"]
        total += traj["tokens"][f"answer_{r}"]["total_tokens"]
    return total


def cumulative_wall_proxy_to_round(traj: Dict[str, Any], round_idx: int) -> float:
    total = traj["stage_wall_time"]["answer_0"]
    for r in range(1, round_idx + 1):
        total += traj["stage_wall_time"][f"feedback_{r}"]
        total += traj["stage_wall_time"][f"answer_{r}"]
    return total


def evaluate_trajectory(traj: Dict[str, Any]) -> Dict[str, Any]:
    fixed = {}
    correctness = {}

    for r in range(0, MAX_ROUNDS + 1):
        correct, evaluator = is_correct(traj["answers"][str(r)], traj["solution"])
        correctness[r] = correct
        fixed[str(r)] = {
            "correct": correct,
            "evaluator": evaluator,
            "tokens": cumulative_tokens_to_round(traj, r),
            "wall_proxy_seconds": cumulative_wall_proxy_to_round(traj, r),
        }

    stop_round = MAX_ROUNDS
    stop_reason = "max_rounds"
    stability_checks = {}

    for r in range(1, MAX_ROUNDS + 1):
        stable, method = semantically_equivalent(
            traj["answers"][str(r - 1)],
            traj["answers"][str(r)],
        )
        stability_checks[str(r)] = {"stable": stable, "method": method}
        if stable:
            stop_round = r
            stop_reason = "answer_stable"
            break

    adaptive_correct = correctness[stop_round]

    transitions = []
    for r in range(1, stop_round + 1):
        prev = correctness[r - 1]
        cur = correctness[r]
        if (not prev) and cur:
            transition = "wrong_to_correct"
        elif prev and (not cur):
            transition = "correct_to_wrong"
        elif prev and cur:
            transition = "correct_to_correct"
        else:
            transition = "wrong_to_wrong"
        transitions.append({"round": r, "transition": transition})

    premature_stop = False
    if stop_round < MAX_ROUNDS and not adaptive_correct:
        premature_stop = any(correctness[r] for r in range(stop_round + 1, MAX_ROUNDS + 1))

    avoided_degradation = False
    if stop_round < MAX_ROUNDS and adaptive_correct:
        avoided_degradation = any(not correctness[r] for r in range(stop_round + 1, MAX_ROUNDS + 1))

    return {
        "question_id": traj["question_id"],
        "level": traj["level"],
        "repetition": traj["repetition"],
        "fixed": fixed,
        "adaptive": {
            "stop_round": stop_round,
            "stop_reason": stop_reason,
            "correct": adaptive_correct,
            "tokens": cumulative_tokens_to_round(traj, stop_round),
            "wall_proxy_seconds": cumulative_wall_proxy_to_round(traj, stop_round),
            "stability_checks": stability_checks,
            "transitions": transitions,
            "premature_stop": premature_stop,
            "avoided_later_degradation": avoided_degradation,
        },
        "answers": traj["answers"],
        "feedback": traj["feedback"],
        "tokens_by_stage": traj["tokens"],
    }


def summarize_repetition(rows: List[Dict[str, Any]], repetition: int) -> Dict[str, Any]:
    n = len(rows)
    result: Dict[str, Any] = {"repetition": repetition, "questions": n, "fixed": {}}

    for r in range(0, MAX_ROUNDS + 1):
        key = str(r)
        acc = mean(1.0 if row["fixed"][key]["correct"] else 0.0 for row in rows)
        tokens = mean(row["fixed"][key]["tokens"] for row in rows)
        result["fixed"][key] = {
            "round": r,
            "accuracy": acc,
            "mean_tokens_per_question": tokens,
        }

    adaptive_acc = mean(1.0 if row["adaptive"]["correct"] else 0.0 for row in rows)
    adaptive_tokens = mean(row["adaptive"]["tokens"] for row in rows)
    adaptive_rounds = mean(row["adaptive"]["stop_round"] for row in rows)

    stop_counts = Counter(row["adaptive"]["stop_round"] for row in rows)
    premature = sum(row["adaptive"]["premature_stop"] for row in rows)
    avoided = sum(row["adaptive"]["avoided_later_degradation"] for row in rows)

    transitions = Counter()
    for row in rows:
        for t in row["adaptive"]["transitions"]:
            transitions[t["transition"]] += 1

    by_level = {}
    for level in LEVELS:
        subset = [row for row in rows if row["level"] == level]
        by_level[str(level)] = {
            "n": len(subset),
            "accuracy": mean(1.0 if row["adaptive"]["correct"] else 0.0 for row in subset),
            "mean_rounds": mean(row["adaptive"]["stop_round"] for row in subset),
            "mean_tokens_per_question": mean(row["adaptive"]["tokens"] for row in subset),
        }

    fixed_r2_tokens = result["fixed"]["2"]["mean_tokens_per_question"]
    fixed_r3_tokens = result["fixed"]["3"]["mean_tokens_per_question"]

    result["adaptive"] = {
        "accuracy": adaptive_acc,
        "mean_rounds": adaptive_rounds,
        "mean_tokens_per_question": adaptive_tokens,
        "stop_distribution": {
            str(r): {
                "count": stop_counts.get(r, 0),
                "fraction": stop_counts.get(r, 0) / n,
            }
            for r in range(1, MAX_ROUNDS + 1)
        },
        "token_savings_vs_fixed_r2_fraction": (
            (fixed_r2_tokens - adaptive_tokens) / fixed_r2_tokens if fixed_r2_tokens else None
        ),
        "token_savings_vs_fixed_r3_fraction": (
            (fixed_r3_tokens - adaptive_tokens) / fixed_r3_tokens if fixed_r3_tokens else None
        ),
        "premature_stop_count": premature,
        "premature_stop_rate": premature / n,
        "avoided_later_degradation_count": avoided,
        "avoided_later_degradation_rate": avoided / n,
        "transitions": dict(transitions),
        "by_level": by_level,
    }
    return result


def aggregate_summaries(rep_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    final: Dict[str, Any] = {
        "experiment": "D - Adaptive Self-Refine",
        "goal": (
            "Test whether semantic answer-stability stopping preserves most fixed-round "
            "Self-Refine accuracy while reducing interaction cost."
        ),
        "questions_per_repetition": rep_summaries[0]["questions"] if rep_summaries else 0,
        "repetitions": len(rep_summaries),
        "max_rounds": MAX_ROUNDS,
        "policy": (
            "Always run one Self-Refine round; stop when consecutive final answers are "
            "semantically equivalent, otherwise continue to R3."
        ),
        "fixed": {},
        "adaptive": {},
    }

    for r in range(0, MAX_ROUNDS + 1):
        values_acc = [s["fixed"][str(r)]["accuracy"] for s in rep_summaries]
        values_tok = [s["fixed"][str(r)]["mean_tokens_per_question"] for s in rep_summaries]
        final["fixed"][str(r)] = {
            "round": r,
            "accuracy_mean": mean(values_acc),
            "accuracy_std_population": pstdev(values_acc) if len(values_acc) > 1 else 0.0,
            "accuracy_values": values_acc,
            "mean_tokens_per_question": mean(values_tok),
        }

    adaptive_acc = [s["adaptive"]["accuracy"] for s in rep_summaries]
    adaptive_rounds = [s["adaptive"]["mean_rounds"] for s in rep_summaries]
    adaptive_tokens = [s["adaptive"]["mean_tokens_per_question"] for s in rep_summaries]
    savings_r2 = [s["adaptive"]["token_savings_vs_fixed_r2_fraction"] for s in rep_summaries]
    savings_r3 = [s["adaptive"]["token_savings_vs_fixed_r3_fraction"] for s in rep_summaries]
    premature = [s["adaptive"]["premature_stop_rate"] for s in rep_summaries]
    avoided = [s["adaptive"]["avoided_later_degradation_rate"] for s in rep_summaries]

    final["adaptive"] = {
        "accuracy_mean": mean(adaptive_acc),
        "accuracy_std_population": pstdev(adaptive_acc) if len(adaptive_acc) > 1 else 0.0,
        "accuracy_values": adaptive_acc,
        "mean_rounds": mean(adaptive_rounds),
        "mean_tokens_per_question": mean(adaptive_tokens),
        "token_savings_vs_fixed_r2_fraction": mean(savings_r2),
        "token_savings_vs_fixed_r3_fraction": mean(savings_r3),
        "premature_stop_rate": mean(premature),
        "avoided_later_degradation_rate": mean(avoided),
        "stop_distribution_mean_fraction": {
            str(r): mean(s["adaptive"]["stop_distribution"][str(r)]["fraction"] for s in rep_summaries)
            for r in range(1, MAX_ROUNDS + 1)
        },
        "by_level_mean": {
            str(level): {
                "accuracy": mean(s["adaptive"]["by_level"][str(level)]["accuracy"] for s in rep_summaries),
                "mean_rounds": mean(s["adaptive"]["by_level"][str(level)]["mean_rounds"] for s in rep_summaries),
                "mean_tokens_per_question": mean(
                    s["adaptive"]["by_level"][str(level)]["mean_tokens_per_question"]
                    for s in rep_summaries
                ),
            }
            for level in LEVELS
        },
    }
    return final


def write_comparison_csv(final: Dict[str, Any], path: Path) -> None:
    rows = []
    r3_tokens = final["fixed"]["3"]["mean_tokens_per_question"]
    for r in range(0, MAX_ROUNDS + 1):
        item = final["fixed"][str(r)]
        rows.append({
            "strategy": f"Fixed R{r}" if r > 0 else "Single / R0",
            "accuracy": item["accuracy_mean"],
            "mean_tokens_per_question": item["mean_tokens_per_question"],
            "mean_rounds": r,
            "token_savings_vs_fixed_r3": 1 - item["mean_tokens_per_question"] / r3_tokens,
        })
    rows.append({
        "strategy": "Adaptive",
        "accuracy": final["adaptive"]["accuracy_mean"],
        "mean_tokens_per_question": final["adaptive"]["mean_tokens_per_question"],
        "mean_rounds": final["adaptive"]["mean_rounds"],
        "token_savings_vs_fixed_r3": final["adaptive"]["token_savings_vs_fixed_r3_fraction"],
    })

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_quick_read(final: Dict[str, Any], path: Path) -> None:
    lines = ["EXPERIMENT D - ADAPTIVE SELF-REFINE", "", "Fixed-round baselines:"]
    for r in range(0, MAX_ROUNDS + 1):
        x = final["fixed"][str(r)]
        lines.append(
            f"  R{r}: accuracy={100*x['accuracy_mean']:.2f}% | "
            f"tokens/q={x['mean_tokens_per_question']:.1f}"
        )

    a = final["adaptive"]
    lines.extend([
        "",
        "Adaptive stopping:",
        f"  accuracy={100*a['accuracy_mean']:.2f}%",
        f"  mean rounds={a['mean_rounds']:.3f}",
        f"  tokens/q={a['mean_tokens_per_question']:.1f}",
        f"  token savings vs R2={100*a['token_savings_vs_fixed_r2_fraction']:.2f}%",
        f"  token savings vs R3={100*a['token_savings_vs_fixed_r3_fraction']:.2f}%",
        f"  premature-stop rate={100*a['premature_stop_rate']:.2f}%",
        f"  avoided-later-degradation rate={100*a['avoided_later_degradation_rate']:.2f}%",
        "",
        "Mean stopping distribution:",
    ])
    for r in range(1, MAX_ROUNDS + 1):
        lines.append(f"  stop at R{r}: {100*a['stop_distribution_mean_fraction'][str(r)]:.2f}%")

    lines.extend(["", "Adaptive mean rounds by MATH level:"])
    for level in LEVELS:
        x = a["by_level_mean"][str(level)]
        lines.append(
            f"  Level {level}: rounds={x['mean_rounds']:.3f} | "
            f"accuracy={100*x['accuracy']:.2f}% | tokens/q={x['mean_tokens_per_question']:.1f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_actual_adaptive_runtime(
    llm: LLM,
    tokenizer,
    questions: List[Dict[str, Any]],
    repetition: int,
) -> Dict[str, Any]:
    """Truly adaptive run used only to validate actual runtime behavior."""
    seed_base = BASE_GENERATION_SEED + 50000 + 1000 * repetition
    n = len(questions)
    states = {
        i: {"answer_prev": None, "answer_cur": None, "stop_round": None, "tokens": 0}
        for i in range(n)
    }

    start_total = time.perf_counter()

    initial_prompts = [
        make_chat(tokenizer, f"{q['problem']}\n\nSolve the problem.")
        for q in questions
    ]
    answers, stats, _ = generate_batch(llm, initial_prompts, MAX_TOKENS_ANSWER, seed_base)
    for i in range(n):
        states[i]["answer_prev"] = answers[i]
        states[i]["tokens"] += stats[i]["total_tokens"]

    active = list(range(n))
    for r in range(1, MAX_ROUNDS + 1):
        if not active:
            break

        fb_prompts = [
            make_chat(
                tokenizer,
                SELF_FEEDBACK_PROMPT.format(
                    problem=questions[i]["problem"],
                    answer=states[i]["answer_prev"],
                ),
            )
            for i in active
        ]
        feedback, fb_stats, _ = generate_batch(
            llm, fb_prompts, MAX_TOKENS_FEEDBACK, seed_base + 10 * r + 1
        )
        for j, i in enumerate(active):
            states[i]["tokens"] += fb_stats[j]["total_tokens"]

        rev_prompts = [
            make_chat(
                tokenizer,
                SELF_REVISION_PROMPT.format(
                    problem=questions[i]["problem"],
                    answer=states[i]["answer_prev"],
                    feedback=feedback[j],
                ),
            )
            for j, i in enumerate(active)
        ]
        revised, rev_stats, _ = generate_batch(
            llm, rev_prompts, MAX_TOKENS_REVISION, seed_base + 10 * r + 2
        )

        next_active = []
        for j, i in enumerate(active):
            states[i]["tokens"] += rev_stats[j]["total_tokens"]
            states[i]["answer_cur"] = revised[j]
            stable, _ = semantically_equivalent(states[i]["answer_prev"], states[i]["answer_cur"])
            if stable or r == MAX_ROUNDS:
                states[i]["stop_round"] = r
            else:
                states[i]["answer_prev"] = states[i]["answer_cur"]
                next_active.append(i)
        active = next_active

    wall_total = time.perf_counter() - start_total

    correctness = []
    rounds = []
    tokens = []
    for i in range(n):
        final_answer = states[i]["answer_cur"] if states[i]["answer_cur"] is not None else states[i]["answer_prev"]
        ok, _ = is_correct(final_answer, questions[i]["solution"])
        correctness.append(ok)
        rounds.append(states[i]["stop_round"] or MAX_ROUNDS)
        tokens.append(states[i]["tokens"])

    return {
        "repetition": repetition,
        "questions": n,
        "accuracy": mean(1.0 if x else 0.0 for x in correctness),
        "mean_rounds": mean(rounds),
        "mean_tokens_per_question": mean(tokens),
        "actual_wall_time_seconds": wall_total,
        "stop_distribution": {
            str(r): {"count": rounds.count(r), "fraction": rounds.count(r) / n}
            for r in range(1, MAX_ROUNDS + 1)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-validation",
        action="store_true",
        help="Also perform a second truly adaptive execution to measure actual wall time.",
    )
    parser.add_argument("--repetitions", type=int, default=N_REPETITIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_or_create_sample(RESULT_DIR)
    print(f"Loaded {len(questions)} questions.")
    print(Counter(q["level"] for q in questions))

    print(f"Loading model: {MODEL_PATH}")
    llm = LLM(
        model=str(MODEL_PATH),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        seed=BASE_GENERATION_SEED,
    )
    tokenizer = llm.get_tokenizer()

    raw_path = RESULT_DIR / "raw_results.jsonl"
    if raw_path.exists():
        raw_path.unlink()

    rep_summaries = []

    for rep in range(1, args.repetitions + 1):
        print(f"\n===== Repetition {rep}/{args.repetitions} =====")

        trajectories, timing = run_full_trajectory(llm, tokenizer, questions, rep)
        evaluated = [evaluate_trajectory(t) for t in trajectories]
        append_jsonl(raw_path, evaluated)

        summary = summarize_repetition(evaluated, rep)
        summary["timing_note"] = (
            "Primary fixed/adaptive results are paired on the same full R3 trajectories. "
            "Adaptive token counts are counterfactual consumed tokens. Wall proxies are "
            "not measured adaptive runtime."
        )
        summary.update(timing)
        dump_json(RESULT_DIR / f"repetition_{rep}_summary.json", summary)
        rep_summaries.append(summary)

        print(
            f"R0={100*summary['fixed']['0']['accuracy']:.2f}% | "
            f"R1={100*summary['fixed']['1']['accuracy']:.2f}% | "
            f"R2={100*summary['fixed']['2']['accuracy']:.2f}% | "
            f"R3={100*summary['fixed']['3']['accuracy']:.2f}% | "
            f"Adaptive={100*summary['adaptive']['accuracy']:.2f}% | "
            f"mean rounds={summary['adaptive']['mean_rounds']:.3f}"
        )

        if args.runtime_validation:
            print("Running actual adaptive runtime validation...")
            runtime_result = run_actual_adaptive_runtime(llm, tokenizer, questions, rep)
            dump_json(RESULT_DIR / f"runtime_validation_rep_{rep}.json", runtime_result)
            print(
                f"Runtime adaptive: accuracy={100*runtime_result['accuracy']:.2f}% | "
                f"rounds={runtime_result['mean_rounds']:.3f} | "
                f"tokens/q={runtime_result['mean_tokens_per_question']:.1f} | "
                f"wall={runtime_result['actual_wall_time_seconds']:.1f}s"
            )

    final = aggregate_summaries(rep_summaries)
    final["math_verify_available"] = MATH_VERIFY_AVAILABLE
    final["dataset"] = DATASET_NAME
    final["model"] = str(MODEL_PATH)
    final["sample_seed"] = BASE_SAMPLE_SEED
    final["generation_seed_base"] = BASE_GENERATION_SEED

    dump_json(RESULT_DIR / "final_summary.json", final)
    write_comparison_csv(final, RESULT_DIR / "comparison_summary.csv")
    write_quick_read(final, RESULT_DIR / "quick_read.txt")

    print("\nExperiment D complete.")
    print(f"Results: {RESULT_DIR}")
    print((RESULT_DIR / "quick_read.txt").read_text())


if __name__ == "__main__":
    main()
