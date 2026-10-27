#!/usr/bin/env python3

import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from math_verify import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        parse as math_parse,
        verify as math_verify,
    )
except ImportError as exc:
    raise SystemExit("Install math-verify with: pip install 'math-verify[antlr4_13_2]'") from exc


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/lustre06/project/6087921/isa1909"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH", str(PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct")
)
RAW_RESULTS_B = Path(
    os.environ.get(
        "RAW_RESULTS_B",
        str(PROJECT_ROOT / "results" / "experiment_7_round_scaling_difficulty" / "raw_results.jsonl"),
    )
)
RESULTS_DIR = Path(
    os.environ.get(
        "RESULTS_DIR", str(PROJECT_ROOT / "results" / "experiment_B2_protocol_validation")
    )
)

N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "300"))
N_REPETITIONS = int(os.environ.get("N_REPETITIONS", "3"))
MAX_CRITIC_ROUNDS = int(os.environ.get("MAX_CRITIC_ROUNDS", "3"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "64"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
GENERATION_SEED = int(os.environ.get("GENERATION_SEED", "42"))
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", "20261005"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_RE = re.compile(r"FINAL:\s*(.+?)(?:\n|$)", re.I)
VERDICT_RE = re.compile(r"VERDICT:\s*(ACCEPT|REJECT)", re.I)


tokenizer = None
llm = None


# ============================================================
# PROMPTS
# ============================================================

SOLVER_SYSTEM = """
You are a careful competition mathematics solver.
Solve independently and show enough reasoning to make the solution checkable.
At the end output exactly: FINAL: <answer>
""".strip()

# Exact Critic prompt used in Experiment B / Experiment 7.
CURRENT_CRITIC_SYSTEM = """
You are an independent mathematical Critic.
First determine the mathematical requirements of the problem, then inspect the Student solution step by step.
Do not reject because you prefer another method.
Reject only if you can identify a concrete mathematical error.

Return exactly:
VERDICT: ACCEPT or REJECT
ERROR_STEP: <short description or NONE>
CRITIQUE: <concise explanation>
SUGGESTION: <actionable correction or NONE>
""".strip()

CURRENT_STUDENT_UPDATE_SYSTEM = """
You are the Student revising after Critic feedback.
Check the Critic independently because it may be wrong.
If VERDICT is ACCEPT, preserve the previous reasoning and final answer unless you independently find a concrete error.
If VERDICT is REJECT, modify only what is supported by a specific identified error.
End with exactly: FINAL: <answer>
""".strip()

# Stronger evidence gate. This is deliberately a minimal change to the B protocol.
EVIDENCE_GATED_CRITIC_SYSTEM = """
You are an independent mathematical Critic.
Inspect the Student solution step by step.

Use VERDICT: REJECT only when you can identify a specific mathematical error and provide concrete evidence showing why that step is invalid or what the corrected local result should be.
If the solution is correct, if the issue is only stylistic, or if you are uncertain whether a real mathematical error exists, use VERDICT: ACCEPT.
Do not reject because you prefer another method.

Return exactly:
VERDICT: ACCEPT or REJECT
ERROR_STEP: <specific erroneous step or NONE>
EVIDENCE: <concrete mathematical evidence/correction or NONE>
SUGGESTION: <minimal actionable correction or NONE>
""".strip()

EVIDENCE_GATED_STUDENT_UPDATE_SYSTEM = """
You are the Student revising after Critic feedback.
Treat the Critic as fallible.
Preserve the current solution unless the feedback contains both VERDICT: REJECT and a concrete mathematical error supported by specific evidence.
If that evidence is valid, make only the minimal necessary correction.
If the evidence is missing, vague, stylistic, or not independently convincing, keep the current reasoning and final answer.
End with exactly: FINAL: <answer>
""".strip()

# Exact debate and judge prompts used in Experiment B / Experiment 7.
DEBATE_SYSTEM = """
You are one solver in a multi-agent mathematical debate.
Read the other agent's solution and compare it with your own current solution.
Keep correct reasoning. Revise only if the other agent exposes a concrete error or gives a stronger mathematical argument.
Do not agree only to create consensus.
End with exactly: FINAL: <answer>
""".strip()

JUDGE_SYSTEM = """
You are the final judge in a mathematical debate.
Given the problem and two agent solutions, check both independently and select the mathematically best answer.
Agreement between agents is not evidence by itself; verify the reasoning.
End with exactly: FINAL: <answer>
""".strip()


# ============================================================
# HELPERS
# ============================================================

def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_messages(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def solver_messages(problem):
    return build_messages(SOLVER_SYSTEM, problem)


def critic_messages(problem, previous_solution, variant):
    system = CURRENT_CRITIC_SYSTEM if variant == "current" else EVIDENCE_GATED_CRITIC_SYSTEM
    return build_messages(
        system,
        f"ORIGINAL PROBLEM\n{problem}\n\nSTUDENT SOLUTION\n{previous_solution}",
    )


def student_update_messages(problem, previous_solution, critique, variant):
    system = (
        CURRENT_STUDENT_UPDATE_SYSTEM
        if variant == "current"
        else EVIDENCE_GATED_STUDENT_UPDATE_SYSTEM
    )
    return build_messages(
        system,
        f"ORIGINAL PROBLEM\n{problem}\n\nCURRENT STUDENT SOLUTION\n{previous_solution}"
        f"\n\nCRITIC FEEDBACK\n{critique}",
    )


def debate_messages(problem, own_solution, other_solution, role):
    return build_messages(
        DEBATE_SYSTEM,
        f"ORIGINAL PROBLEM\n{problem}\n\nYOUR CURRENT SOLUTION ({role})\n{own_solution}"
        f"\n\nOTHER AGENT SOLUTION\n{other_solution}",
    )


def judge_messages(problem, agent_a, agent_b):
    return build_messages(
        JUDGE_SYSTEM,
        f"ORIGINAL PROBLEM\n{problem}\n\nAGENT A SOLUTION\n{agent_a}"
        f"\n\nAGENT B SOLUTION\n{agent_b}",
    )


def extract_final_text(text):
    if not text:
        return None
    matches = FINAL_RE.findall(text)
    return matches[-1].strip() if matches else text.strip()


def parse_math(text):
    if text is None:
        return []
    value = str(text).strip()
    if not ((value.startswith("$") and value.endswith("$")) or "\\boxed" in value):
        value = f"${value}$"
    return math_parse(
        value,
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
    )


def answers_match(prediction, ground_truth):
    if prediction is None:
        return False
    try:
        pred = parse_math(prediction)
        gold = parse_math(ground_truth)
        return bool(pred and gold and math_verify(gold, pred))
    except Exception:
        return False


def parse_verdict(text):
    match = VERDICT_RE.search(text or "")
    return match.group(1).upper() if match else "UNKNOWN"


def correction_counts(before, after):
    positive = sum((not b) and a for b, a in zip(before, after))
    negative = sum(b and (not a) for b, a in zip(before, after))
    wrong = sum(not b for b in before)
    right = sum(b for b in before)
    return {
        "positive_corrections": positive,
        "negative_corrections": negative,
        "positive_correction_rate": positive / wrong if wrong else None,
        "negative_correction_rate": negative / right if right else None,
    }


def critic_quality(verdicts, correct_before):
    tr = fr = ta = fa = unknown = 0
    for verdict, correct in zip(verdicts, correct_before):
        if verdict == "UNKNOWN":
            unknown += 1
        elif verdict == "REJECT" and not correct:
            tr += 1
        elif verdict == "REJECT" and correct:
            fr += 1
        elif verdict == "ACCEPT" and correct:
            ta += 1
        elif verdict == "ACCEPT" and not correct:
            fa += 1
    precision = tr / (tr + fr) if tr + fr else None
    recall = tr / (tr + fa) if tr + fa else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "true_reject": tr,
        "false_reject": fr,
        "true_accept": ta,
        "false_accept": fa,
        "unknown_verdict": unknown,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reject_rate": (tr + fr) / len(verdicts) if verdicts else None,
    }


# ============================================================
# DATA: REUSE QUESTIONS FROM EXPERIMENT B
# ============================================================

def load_questions_from_experiment_b():
    if not RAW_RESULTS_B.exists():
        raise FileNotFoundError(f"Experiment B trace not found: {RAW_RESULTS_B}")

    # Read one copy of each question from repetition 1.
    candidates = {}
    with open(RAW_RESULTS_B, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if int(row.get("repetition", -1)) != 1:
                continue
            qid = int(row["question_id"])
            candidates[qid] = {
                "question_id_B": qid,
                "dataset_index": int(row["dataset_index"]),
                "level": row.get("level"),
            }

    if len(candidates) < N_QUESTIONS:
        raise ValueError(
            f"Experiment B contains only {len(candidates)} unique questions; "
            f"N_QUESTIONS={N_QUESTIONS}."
        )

    # Balanced deterministic subset across MATH levels, drawn only from Experiment B questions.
    by_level = defaultdict(list)
    for item in candidates.values():
        if item["level"] in {1, 2, 3, 4, 5}:
            by_level[int(item["level"])].append(item)

    rng = random.Random(SAMPLE_SEED)
    for level in by_level:
        rng.shuffle(by_level[level])

    base = N_QUESTIONS // 5
    remainder = N_QUESTIONS % 5
    selected = []
    for level in range(1, 6):
        need = base + (1 if level <= remainder else 0)
        if len(by_level[level]) < need:
            raise ValueError(f"Not enough Experiment B questions at level {level}: need {need}")
        selected.extend(by_level[level][:need])

    selected.sort(key=lambda x: (x["level"], x["question_id_B"]))

    dataset = load_dataset("jeggers/competition_math", "original", split="test")
    questions = []
    for new_id, item in enumerate(selected, start=1):
        row = dataset[item["dataset_index"]]
        gold = row.get("extracted_solution")
        if gold is None:
            raise ValueError(f"Dataset row {item['dataset_index']} has no extracted_solution")
        questions.append(
            {
                "question_id": new_id,
                "question_id_B": item["question_id_B"],
                "dataset_index": item["dataset_index"],
                "problem": row["problem"],
                "gold": str(gold),
                "level": int(item["level"]),
                "type": row.get("type"),
            }
        )

    counts = {level: sum(q["level"] == level for q in questions) for level in range(1, 6)}
    print("B2 sample from Experiment B, by MATH level:", counts, flush=True)
    return questions


# ============================================================
# VLLM
# ============================================================

def initialize_engine():
    global tokenizer, llm
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )


def render_messages(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    if prompt_tokens + MAX_NEW_TOKENS > MAX_MODEL_LEN:
        raise ValueError(f"Prompt too long: {prompt_tokens}+{MAX_NEW_TOKENS}>{MAX_MODEL_LEN}")
    return prompt


def generate_batch(messages_batch, seed):
    prompts = [render_messages(messages) for messages in messages_batch]
    params = SamplingParams(
        n=1,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        seed=seed,
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=True)
    wall = time.perf_counter() - start

    parsed = []
    for request_output in outputs:
        prompt_tokens = len(getattr(request_output, "prompt_token_ids", []) or [])
        candidate = request_output.outputs[0]
        completion_tokens = len(candidate.token_ids)
        parsed.append(
            {
                "text": candidate.text,
                "answer": extract_final_text(candidate.text),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        )
    return parsed, wall


# ============================================================
# EXPERIMENT
# ============================================================

def run_critic_variant(questions, initial, initial_correct, initial_tokens, initial_time, rep_seed, variant):
    n = len(questions)
    current = initial
    before_correct = initial_correct
    cumulative_tokens = initial_tokens.copy()
    cumulative_time = initial_time
    rounds = []
    traces = {i: [] for i in range(n)}

    variant_offset = 0 if variant == "current" else 1000

    for round_id in range(1, MAX_CRITIC_ROUNDS + 1):
        critiques, dt = generate_batch(
            [critic_messages(questions[i]["problem"], current[i]["text"], variant) for i in range(n)],
            rep_seed + variant_offset + 100 + 2 * round_id,
        )
        cumulative_time += dt
        verdicts = [parse_verdict(x["text"]) for x in critiques]

        updated, dt = generate_batch(
            [
                student_update_messages(
                    questions[i]["problem"], current[i]["text"], critiques[i]["text"], variant
                )
                for i in range(n)
            ],
            rep_seed + variant_offset + 101 + 2 * round_id,
        )
        cumulative_time += dt

        after_correct = [
            answers_match(updated[i]["answer"], questions[i]["gold"]) for i in range(n)
        ]
        cumulative_tokens = [
            cumulative_tokens[i] + critiques[i]["total_tokens"] + updated[i]["total_tokens"]
            for i in range(n)
        ]

        metrics = {
            "variant": variant,
            "round": round_id,
            "accuracy": sum(after_correct) / n,
            "critic": critic_quality(verdicts, before_correct),
            "transitions": correction_counts(before_correct, after_correct),
            "total_tokens": sum(cumulative_tokens),
            "mean_tokens_per_question": mean(cumulative_tokens),
            "cumulative_wall_time_seconds": cumulative_time,
        }
        rounds.append(metrics)

        for i in range(n):
            traces[i].append(
                {
                    "round": round_id,
                    "before": current[i],
                    "critique": critiques[i],
                    "verdict": verdicts[i],
                    "after": updated[i],
                    "correct_before": before_correct[i],
                    "correct_after": after_correct[i],
                }
            )

        current = updated
        before_correct = after_correct

    return rounds, traces


def judge_diagnostics(agent_a_correct, agent_b_correct, judge_correct, disagreement):
    n = len(judge_correct)
    exactly_one = [a != b for a, b in zip(agent_a_correct, agent_b_correct)]
    both_correct = [a and b for a, b in zip(agent_a_correct, agent_b_correct)]
    both_wrong = [(not a) and (not b) for a, b in zip(agent_a_correct, agent_b_correct)]

    def conditional_rate(mask, outcome):
        idx = [i for i, flag in enumerate(mask) if flag]
        return sum(outcome[i] for i in idx) / len(idx) if idx else None

    return {
        "judge_accuracy": sum(judge_correct) / n,
        "disagreement_rate": sum(disagreement) / n,
        "exactly_one_agent_correct_cases": sum(exactly_one),
        "judge_correct_when_exactly_one_agent_correct": conditional_rate(exactly_one, judge_correct),
        "both_agents_correct_cases": sum(both_correct),
        "judge_error_when_both_agents_correct": (
            1 - conditional_rate(both_correct, judge_correct)
            if sum(both_correct) and conditional_rate(both_correct, judge_correct) is not None
            else None
        ),
        "both_agents_wrong_cases": sum(both_wrong),
        "judge_rescue_when_both_agents_wrong": conditional_rate(both_wrong, judge_correct),
    }


def run_debate_validation(questions, initial, initial_correct, initial_tokens, initial_time, rep_seed):
    n = len(questions)

    b0, b0_time = generate_batch(
        [solver_messages(q["problem"]) for q in questions], rep_seed + 2000
    )
    b0_correct = [answers_match(b0[i]["answer"], questions[i]["gold"]) for i in range(n)]
    disagreement0 = [not answers_match(initial[i]["answer"], b0[i]["answer"]) for i in range(n)]

    judge0, judge0_time = generate_batch(
        [judge_messages(questions[i]["problem"], initial[i]["text"], b0[i]["text"]) for i in range(n)],
        rep_seed + 2100,
    )
    judge0_correct = [answers_match(judge0[i]["answer"], questions[i]["gold"]) for i in range(n)]

    a1, a1_time = generate_batch(
        [
            debate_messages(questions[i]["problem"], initial[i]["text"], b0[i]["text"], "Agent A")
            for i in range(n)
        ],
        rep_seed + 2200,
    )
    b1, b1_time = generate_batch(
        [
            debate_messages(questions[i]["problem"], b0[i]["text"], initial[i]["text"], "Agent B")
            for i in range(n)
        ],
        rep_seed + 2201,
    )
    a1_correct = [answers_match(a1[i]["answer"], questions[i]["gold"]) for i in range(n)]
    b1_correct = [answers_match(b1[i]["answer"], questions[i]["gold"]) for i in range(n)]
    disagreement1 = [not answers_match(a1[i]["answer"], b1[i]["answer"]) for i in range(n)]

    judge1, judge1_time = generate_batch(
        [judge_messages(questions[i]["problem"], a1[i]["text"], b1[i]["text"]) for i in range(n)],
        rep_seed + 2300,
    )
    judge1_correct = [answers_match(judge1[i]["answer"], questions[i]["gold"]) for i in range(n)]

    consensus_idx = [i for i, disagree in enumerate(disagreement1) if not disagree]
    consensus_accuracy = (
        sum(a1_correct[i] for i in consensus_idx) / len(consensus_idx) if consensus_idx else None
    )
    wrong_consensus_rate = (
        sum(not a1_correct[i] for i in consensus_idx) / len(consensus_idx) if consensus_idx else None
    )

    tokens_r0 = [initial_tokens[i] + b0[i]["total_tokens"] + judge0[i]["total_tokens"] for i in range(n)]
    tokens_r1 = [
        initial_tokens[i]
        + b0[i]["total_tokens"]
        + a1[i]["total_tokens"]
        + b1[i]["total_tokens"]
        + judge1[i]["total_tokens"]
        for i in range(n)
    ]

    summary = {
        "no_interaction_judge": {
            "accuracy": sum(judge0_correct) / n,
            "total_tokens": sum(tokens_r0),
            "mean_tokens_per_question": mean(tokens_r0),
            "wall_time_seconds": initial_time + b0_time + judge0_time,
            "diagnostics": judge_diagnostics(initial_correct, b0_correct, judge0_correct, disagreement0),
        },
        "one_interaction_plus_judge": {
            "accuracy": sum(judge1_correct) / n,
            "total_tokens": sum(tokens_r1),
            "mean_tokens_per_question": mean(tokens_r1),
            "wall_time_seconds": initial_time + b0_time + a1_time + b1_time + judge1_time,
            "transition_vs_no_interaction_judge": correction_counts(judge0_correct, judge1_correct),
            "diagnostics": judge_diagnostics(a1_correct, b1_correct, judge1_correct, disagreement1),
        },
        "interaction_without_judge_diagnostic": {
            "consensus_coverage": len(consensus_idx) / n,
            "consensus_accuracy": consensus_accuracy,
            "wrong_consensus_rate": wrong_consensus_rate,
            "agent_a_accuracy": sum(a1_correct) / n,
            "agent_b_accuracy": sum(b1_correct) / n,
            "disagreement_before": sum(disagreement0) / n,
            "disagreement_after": sum(disagreement1) / n,
        },
    }

    traces = []
    for i in range(n):
        traces.append(
            {
                "agent_a0": initial[i],
                "agent_b0": b0[i],
                "judge_r0": judge0[i],
                "agent_a1": a1[i],
                "agent_b1": b1[i],
                "judge_r1": judge1[i],
                "a0_correct": initial_correct[i],
                "b0_correct": b0_correct[i],
                "judge_r0_correct": judge0_correct[i],
                "a1_correct": a1_correct[i],
                "b1_correct": b1_correct[i],
                "judge_r1_correct": judge1_correct[i],
                "disagreement_r0": disagreement0[i],
                "disagreement_r1": disagreement1[i],
            }
        )

    return summary, traces


def run_repetition(questions, repetition):
    rep_seed = GENERATION_SEED + 10000 * repetition
    n = len(questions)
    start = time.perf_counter()

    print(f"\n{'=' * 70}\nB2 REPETITION {repetition}/{N_REPETITIONS}\n{'=' * 70}", flush=True)

    initial, initial_time = generate_batch(
        [solver_messages(q["problem"]) for q in questions], rep_seed + 1
    )
    initial_correct = [
        answers_match(initial[i]["answer"], questions[i]["gold"]) for i in range(n)
    ]
    initial_tokens = [x["total_tokens"] for x in initial]

    print("[1] Current Student-Critic", flush=True)
    current_rounds, current_traces = run_critic_variant(
        questions, initial, initial_correct, initial_tokens, initial_time, rep_seed, "current"
    )

    print("[2] Evidence-gated Student-Critic", flush=True)
    gated_rounds, gated_traces = run_critic_variant(
        questions, initial, initial_correct, initial_tokens, initial_time, rep_seed, "evidence_gated"
    )

    print("[3] Debate / Judge ablation", flush=True)
    debate_summary, debate_traces = run_debate_validation(
        questions, initial, initial_correct, initial_tokens, initial_time, rep_seed
    )

    rows = []
    for i, q in enumerate(questions):
        rows.append(
            {
                "repetition": repetition,
                **q,
                "single": {"output": initial[i], "correct": initial_correct[i]},
                "student_critic_current": current_traces[i],
                "student_critic_evidence_gated": gated_traces[i],
                "debate": debate_traces[i],
            }
        )

    summary = {
        "repetition": repetition,
        "questions": n,
        "single_accuracy": sum(initial_correct) / n,
        "student_critic": {
            "current": current_rounds,
            "evidence_gated": gated_rounds,
        },
        "debate": debate_summary,
        "wall_time_seconds": time.perf_counter() - start,
    }
    return rows, summary


# ============================================================
# AGGREGATION
# ============================================================

def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std_population": None, "values": []}
    return {
        "mean": mean(values),
        "std_population": pstdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def aggregate(summaries):
    critic_rows = []
    for variant in ["current", "evidence_gated"]:
        for round_id in range(1, MAX_CRITIC_ROUNDS + 1):
            records = [
                next(x for x in s["student_critic"][variant] if x["round"] == round_id)
                for s in summaries
            ]
            critic_rows.append(
                {
                    "variant": variant,
                    "round": round_id,
                    "accuracy": mean_std([r["accuracy"] for r in records]),
                    "false_rejects": mean_std([r["critic"]["false_reject"] for r in records]),
                    "precision": mean_std([r["critic"]["precision"] for r in records]),
                    "recall": mean_std([r["critic"]["recall"] for r in records]),
                    "f1": mean_std([r["critic"]["f1"] for r in records]),
                    "positive_corrections": mean_std([r["transitions"]["positive_corrections"] for r in records]),
                    "negative_corrections": mean_std([r["transitions"]["negative_corrections"] for r in records]),
                    "mean_tokens_per_question": mean_std([r["mean_tokens_per_question"] for r in records]),
                }
            )

    debate = {}
    for key in ["no_interaction_judge", "one_interaction_plus_judge"]:
        debate[key] = {
            "accuracy": mean_std([s["debate"][key]["accuracy"] for s in summaries]),
            "mean_tokens_per_question": mean_std(
                [s["debate"][key]["mean_tokens_per_question"] for s in summaries]
            ),
            "disagreement_rate": mean_std(
                [s["debate"][key]["diagnostics"]["disagreement_rate"] for s in summaries]
            ),
            "judge_correct_when_exactly_one_agent_correct": mean_std(
                [
                    s["debate"][key]["diagnostics"]["judge_correct_when_exactly_one_agent_correct"]
                    for s in summaries
                ]
            ),
            "judge_error_when_both_agents_correct": mean_std(
                [
                    s["debate"][key]["diagnostics"]["judge_error_when_both_agents_correct"]
                    for s in summaries
                ]
            ),
            "judge_rescue_when_both_agents_wrong": mean_std(
                [
                    s["debate"][key]["diagnostics"]["judge_rescue_when_both_agents_wrong"]
                    for s in summaries
                ]
            ),
        }

    debate["interaction_without_judge_diagnostic"] = {
        name: mean_std([s["debate"]["interaction_without_judge_diagnostic"][name] for s in summaries])
        for name in [
            "consensus_coverage",
            "consensus_accuracy",
            "wrong_consensus_rate",
            "agent_a_accuracy",
            "agent_b_accuracy",
            "disagreement_before",
            "disagreement_after",
        ]
    }

    return {
        "experiment": "B2 - Protocol Validation",
        "goal": "Test whether harmful Student-Critic and Debate patterns are caused by protocol design rather than adding a new architecture.",
        "questions": N_QUESTIONS,
        "repetitions": N_REPETITIONS,
        "source_questions": "Balanced subset of questions already used in Experiment B / Experiment 7",
        "single_accuracy": mean_std([s["single_accuracy"] for s in summaries]),
        "student_critic": critic_rows,
        "debate": debate,
        "interpretation_rule": {
            "student_critic": "Adopt the evidence-gated variant only if it reduces false rejects / harmful corrections without materially reducing useful corrections or accuracy.",
            "debate": "Use the ablation to separate gains from peer interaction from gains due to the final judge. Increased consensus alone is not treated as improvement.",
        },
    }


def flat_summary_rows(final):
    rows = []
    for item in final["student_critic"]:
        rows.append(
            {
                "family": "student_critic",
                "condition": item["variant"],
                "round": item["round"],
                "accuracy_mean": item["accuracy"]["mean"],
                "false_rejects_mean": item["false_rejects"]["mean"],
                "precision_mean": item["precision"]["mean"],
                "recall_mean": item["recall"]["mean"],
                "positive_corrections_mean": item["positive_corrections"]["mean"],
                "negative_corrections_mean": item["negative_corrections"]["mean"],
                "mean_tokens_per_question": item["mean_tokens_per_question"]["mean"],
            }
        )
    for key in ["no_interaction_judge", "one_interaction_plus_judge"]:
        d = final["debate"][key]
        rows.append(
            {
                "family": "debate",
                "condition": key,
                "round": "",
                "accuracy_mean": d["accuracy"]["mean"],
                "false_rejects_mean": "",
                "precision_mean": "",
                "recall_mean": "",
                "positive_corrections_mean": "",
                "negative_corrections_mean": "",
                "mean_tokens_per_question": d["mean_tokens_per_question"]["mean"],
            }
        )
    return rows


def write_quick_read(final):
    lines = [
        "B2 -- PROTOCOL VALIDATION",
        "=" * 60,
        f"Questions: {N_QUESTIONS} | repetitions: {N_REPETITIONS}",
        "",
        "Student-Critic: current vs evidence-gated critic",
    ]
    for item in final["student_critic"]:
        lines.append(
            f"  {item['variant']} R{item['round']}: "
            f"acc={item['accuracy']['mean']:.2%}, "
            f"false_rejects={item['false_rejects']['mean']:.1f}, "
            f"precision={item['precision']['mean']:.3f}, "
            f"recall={item['recall']['mean']:.3f}, "
            f"+corr={item['positive_corrections']['mean']:.1f}, "
            f"-corr={item['negative_corrections']['mean']:.1f}"
        )
    lines += ["", "Debate / Judge ablation"]
    for key in ["no_interaction_judge", "one_interaction_plus_judge"]:
        d = final["debate"][key]
        lines.append(
            f"  {key}: acc={d['accuracy']['mean']:.2%}, "
            f"tokens/q={d['mean_tokens_per_question']['mean']:.1f}, "
            f"disagreement={d['disagreement_rate']['mean']:.2%}"
        )
    nd = final["debate"]["interaction_without_judge_diagnostic"]
    lines.append(
        f"  after interaction without judge: consensus coverage={nd['consensus_coverage']['mean']:.2%}, "
        f"consensus accuracy={nd['consensus_accuracy']['mean']:.2%}, "
        f"wrong consensus={nd['wrong_consensus_rate']['mean']:.2%}"
    )
    lines += [
        "",
        "Decision rule:",
        "  Do not keep a revised prompt because it performs better once. Keep it only if it consistently reduces the specific failure it was designed to address.",
    ]
    (RESULTS_DIR / "quick_read.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("B2 - Protocol Validation", flush=True)
    print(f"Model: {MODEL_PATH}", flush=True)
    print(f"Experiment B traces: {RAW_RESULTS_B}", flush=True)
    print(f"Results: {RESULTS_DIR}", flush=True)

    questions = load_questions_from_experiment_b()
    initialize_engine()

    raw_path = RESULTS_DIR / "raw_results.jsonl"
    if raw_path.exists():
        raw_path.unlink()

    summaries = []
    for repetition in range(1, N_REPETITIONS + 1):
        rows, summary = run_repetition(questions, repetition)
        append_jsonl(raw_path, rows)
        save_json(RESULTS_DIR / f"repetition_{repetition}_summary.json", summary)
        summaries.append(summary)

    final = aggregate(summaries)
    save_json(RESULTS_DIR / "final_summary.json", final)
    write_csv(RESULTS_DIR / "comparison_summary.csv", flat_summary_rows(final))
    write_quick_read(final)

    print("\nCompleted B2.", flush=True)
    print(f"Final summary: {RESULTS_DIR / 'final_summary.json'}", flush=True)
    print(f"Quick read: {RESULTS_DIR / 'quick_read.txt'}", flush=True)


if __name__ == "__main__":
    main()
