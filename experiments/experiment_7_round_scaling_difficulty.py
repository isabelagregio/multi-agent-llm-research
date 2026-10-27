#!/usr/bin/env python3

import json
import multiprocessing
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
    raise SystemExit(
        "Install math-verify with: pip install 'math-verify[antlr4_13_2]'"
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", "/home/isa1909/projects/def-sigmagao/isa1909")
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH", str(PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct")
)
RESULTS_DIR = PROJECT_ROOT / "results" / os.environ.get(
    "RESULTS_SUBDIR", "experiment_7_round_scaling_difficulty"
)

N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "500"))
BASE_A_QUESTIONS = int(os.environ.get("BASE_A_QUESTIONS", "200"))
BASE_A_SEED = int(os.environ.get("BASE_A_SEED", "42"))
N_REPETITIONS = int(os.environ.get("N_REPETITIONS", "3"))
ROUND_VALUES = sorted(
    {int(x.strip()) for x in os.environ.get("ROUND_VALUES", "0,1,2,3").split(",")}
)
MAX_ROUNDS = max(ROUND_VALUES)

TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "64"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))

DATASET_SEED = int(os.environ.get("DATASET_SEED", "42"))
GENERATION_SEED = int(os.environ.get("GENERATION_SEED", "42"))

if not ROUND_VALUES or ROUND_VALUES[0] != 0:
    raise ValueError("ROUND_VALUES must include 0, e.g. 0,1,2,3")
if any(r < 0 for r in ROUND_VALUES):
    raise ValueError("ROUND_VALUES cannot contain negative values")


tokenizer = None
llm = None

FINAL_RE = re.compile(r"FINAL:\s*(.+?)(?:\n|$)", re.I)
VERDICT_RE = re.compile(r"VERDICT:\s*(ACCEPT|REJECT)", re.I)
LEVEL_RE = re.compile(r"(\d+)")


# ============================================================
# PROMPTS
# ============================================================

SOLVER_SYSTEM = """
You are a careful competition mathematics solver.
Solve independently and show enough reasoning to make the solution checkable.
At the end output exactly: FINAL: <answer>
""".strip()

SELF_FEEDBACK_SYSTEM = """
Review your own previous mathematical solution.
Check arithmetic, algebra, logic, interpretation, and missing cases.
Do not change a correct solution merely to be different.
Give concise actionable feedback.
""".strip()

SELF_REFINE_SYSTEM = """
Revise your previous solution using the self-feedback.
Check the feedback independently.
Preserve correct reasoning and only change what is necessary.
End with exactly: FINAL: <answer>
""".strip()

CRITIC_SYSTEM = """
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

STUDENT_UPDATE_SYSTEM = """
You are the Student revising after Critic feedback.
Check the Critic independently because it may be wrong.
If VERDICT is ACCEPT, preserve the previous reasoning and final answer unless you independently find a concrete error.
If VERDICT is REJECT, modify only what is supported by a specific identified error.
End with exactly: FINAL: <answer>
""".strip()

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
# BASIC HELPERS
# ============================================================

def save_json(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_level(value):
    match = LEVEL_RE.search(str(value or ""))
    return int(match.group(1)) if match else None


def build_messages(system, user):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def solver_messages(problem):
    return build_messages(SOLVER_SYSTEM, problem)


def self_feedback_messages(problem, previous_solution):
    return build_messages(
        SELF_FEEDBACK_SYSTEM,
        f"ORIGINAL PROBLEM\n{problem}\n\nPREVIOUS SOLUTION\n{previous_solution}",
    )


def self_refine_messages(problem, previous_solution, feedback):
    return build_messages(
        SELF_REFINE_SYSTEM,
        f"ORIGINAL PROBLEM\n{problem}\n\nPREVIOUS SOLUTION\n{previous_solution}"
        f"\n\nSELF-FEEDBACK\n{feedback}",
    )


def critic_messages(problem, previous_solution):
    return build_messages(
        CRITIC_SYSTEM,
        f"ORIGINAL PROBLEM\n{problem}\n\nSTUDENT SOLUTION\n{previous_solution}",
    )


def student_update_messages(problem, previous_solution, critique):
    return build_messages(
        STUDENT_UPDATE_SYSTEM,
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


# ============================================================
# DATASET AND MATH EVALUATION
# ============================================================

def load_math_sample():
    """Keep Experiment A's 200-question sample and add new questions stratified by MATH level."""
    dataset = load_dataset("jeggers/competition_math", "original", split="test")
    if N_QUESTIONS > len(dataset):
        raise ValueError(f"N_QUESTIONS={N_QUESTIONS} > dataset={len(dataset)}")
    if BASE_A_QUESTIONS > N_QUESTIONS:
        raise ValueError("BASE_A_QUESTIONS cannot exceed N_QUESTIONS")

    # Reconstruct the exact Experiment A sample: random.Random(42).sample(..., 200).
    base_indices = random.Random(BASE_A_SEED).sample(range(len(dataset)), BASE_A_QUESTIONS)
    selected = list(base_indices)
    selected_set = set(selected)

    if N_QUESTIONS > BASE_A_QUESTIONS:
        levels = {}
        pools = defaultdict(list)
        for idx in range(len(dataset)):
            level = normalize_level(dataset[idx].get("level"))
            levels[idx] = level
            if idx not in selected_set and level in {1, 2, 3, 4, 5}:
                pools[level].append(idx)

        current = {level: sum(levels[idx] == level for idx in selected) for level in range(1, 6)}
        target_base = N_QUESTIONS // 5
        targets = {level: target_base for level in range(1, 6)}
        for level in range(1, N_QUESTIONS % 5 + 1):
            targets[level] += 1

        rng = random.Random(DATASET_SEED)
        for pool in pools.values():
            rng.shuffle(pool)

        # First fill deficits toward an approximately balanced 1--5 distribution.
        remaining = N_QUESTIONS - len(selected)
        for level in range(1, 6):
            need = max(0, targets[level] - current[level])
            take = min(need, len(pools[level]), remaining)
            selected.extend(pools[level][:take])
            pools[level] = pools[level][take:]
            remaining -= take

        # If exact balance is impossible, fill the remainder round-robin across levels.
        while remaining > 0:
            added = False
            for level in range(1, 6):
                if pools[level] and remaining > 0:
                    selected.append(pools[level].pop())
                    remaining -= 1
                    added = True
            if not added:
                raise ValueError("Not enough remaining MATH questions with levels 1--5")

    questions = []
    for question_id, dataset_index in enumerate(selected, start=1):
        row = dataset[dataset_index]
        gold = row.get("extracted_solution")
        if gold is None:
            raise ValueError(f"Dataset row {dataset_index} has no extracted_solution")
        questions.append({
            "question_id": question_id,
            "dataset_index": dataset_index,
            "problem": row["problem"],
            "gold": str(gold),
            "level": normalize_level(row.get("level")),
            "type": row.get("type"),
            "from_experiment_a_sample": question_id <= BASE_A_QUESTIONS,
        })

    counts = {level: sum(q["level"] == level for q in questions) for level in range(1, 6)}
    print("Sample composition by MATH level:", counts, flush=True)
    print(f"Preserved Experiment A questions: {BASE_A_QUESTIONS}/{len(questions)}", flush=True)
    return questions


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


# ============================================================
# VLLM
# ============================================================

def initialize_engine():
    global tokenizer, llm

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
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
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)

    if prompt_tokens + MAX_NEW_TOKENS > MAX_MODEL_LEN:
        raise ValueError(
            f"Prompt too long: {prompt_tokens}+{MAX_NEW_TOKENS}>{MAX_MODEL_LEN}"
        )
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
    wall_time = time.perf_counter() - start

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

    return parsed, wall_time


# ============================================================
# METRICS
# ============================================================

def correction_counts(before_correct, after_correct):
    positive = sum(not before and after for before, after in zip(before_correct, after_correct))
    negative = sum(before and not after for before, after in zip(before_correct, after_correct))
    wrong_before = sum(not value for value in before_correct)
    right_before = sum(before_correct)

    return {
        "positive_corrections": positive,
        "negative_corrections": negative,
        "positive_correction_rate": positive / wrong_before if wrong_before else None,
        "negative_correction_rate": negative / right_before if right_before else None,
    }


def critic_metrics(records):
    tr = fr = ta = fa = unknown = 0

    for record in records:
        verdict = record["verdict"]
        correct = record["correct_before"]

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
        "error_detection_precision": precision,
        "error_detection_recall": recall,
        "error_detection_f1": f1,
    }


def summarize_checkpoint(method, round_id, correct, tokens, calls, wall_time, baseline_correct):
    accuracy = sum(correct) / len(correct)
    baseline_accuracy = sum(baseline_correct) / len(baseline_correct)
    total_tokens = sum(tokens)
    baseline_tokens = None

    return {
        "method": method,
        "round": round_id,
        "accuracy": accuracy,
        "accuracy_gain_vs_single": accuracy - baseline_accuracy,
        "protocol_generation_calls_per_question": calls,
        "protocol_total_tokens": total_tokens,
        "protocol_mean_tokens_per_question": total_tokens / len(tokens),
        "protocol_wall_time_seconds": wall_time,
        "corrections_vs_single": correction_counts(baseline_correct, correct),
    }


def difficulty_summary(questions, method_states, single_correct):
    levels = sorted({q["level"] for q in questions if q["level"] is not None})
    rows = []

    for level in levels:
        indices = [i for i, q in enumerate(questions) if q["level"] == level]
        if not indices:
            continue

        baseline_acc = sum(single_correct[i] for i in indices) / len(indices)
        for method, checkpoints in method_states.items():
            for round_id, state in checkpoints.items():
                correct = state["correct"]
                acc = sum(correct[i] for i in indices) / len(indices)
                rows.append(
                    {
                        "level": level,
                        "questions": len(indices),
                        "method": method,
                        "round": round_id,
                        "accuracy": acc,
                        "single_accuracy": baseline_acc,
                        "gain_vs_single": acc - baseline_acc,
                    }
                )

    return rows


# ============================================================
# ONE REPETITION
# ============================================================

def run_repetition(questions, repetition):
    rep_start = time.perf_counter()
    rep_seed = GENERATION_SEED + 10000 * repetition
    n = len(questions)

    print(f"\n{'=' * 80}\nREPETITION {repetition}/{N_REPETITIONS}\n{'=' * 80}", flush=True)

    # --------------------------------------------------------
    # Shared initial solution: R=0 for Self-Refine and Student-Critic
    # --------------------------------------------------------
    print("[1] Shared initial solutions", flush=True)
    initial, initial_time = generate_batch(
        [solver_messages(q["problem"]) for q in questions],
        rep_seed + 1,
    )
    single_correct = [
        answers_match(initial[i]["answer"], questions[i]["gold"]) for i in range(n)
    ]
    single_tokens = [item["total_tokens"] for item in initial]

    self_states = {
        0: {
            "outputs": initial,
            "correct": single_correct,
            "tokens": single_tokens,
            "cumulative_wall_time": initial_time,
        }
    }
    critic_states = {
        0: {
            "outputs": initial,
            "correct": single_correct,
            "tokens": single_tokens,
            "cumulative_wall_time": initial_time,
        }
    }

    # --------------------------------------------------------
    # Self-Refine: nested trajectory R=1..MAX_ROUNDS
    # --------------------------------------------------------
    self_current = initial
    self_cumulative_tokens = single_tokens.copy()
    self_cumulative_time = initial_time

    for round_id in range(1, MAX_ROUNDS + 1):
        print(f"[2] Self-Refine round {round_id}/{MAX_ROUNDS}", flush=True)
        feedback, duration = generate_batch(
            [
                self_feedback_messages(questions[i]["problem"], self_current[i]["text"])
                for i in range(n)
            ],
            rep_seed + 100 + 2 * round_id,
        )
        self_cumulative_time += duration

        refined, duration = generate_batch(
            [
                self_refine_messages(
                    questions[i]["problem"],
                    self_current[i]["text"],
                    feedback[i]["text"],
                )
                for i in range(n)
            ],
            rep_seed + 101 + 2 * round_id,
        )
        self_cumulative_time += duration

        self_cumulative_tokens = [
            self_cumulative_tokens[i]
            + feedback[i]["total_tokens"]
            + refined[i]["total_tokens"]
            for i in range(n)
        ]
        correct = [
            answers_match(refined[i]["answer"], questions[i]["gold"]) for i in range(n)
        ]
        self_states[round_id] = {
            "outputs": refined,
            "feedback": feedback,
            "correct": correct,
            "tokens": self_cumulative_tokens.copy(),
            "cumulative_wall_time": self_cumulative_time,
        }
        self_current = refined

    # --------------------------------------------------------
    # Student-Critic: nested trajectory R=1..MAX_ROUNDS
    # --------------------------------------------------------
    critic_current = initial
    critic_cumulative_tokens = single_tokens.copy()
    critic_cumulative_time = initial_time
    critic_round_metrics = {}

    for round_id in range(1, MAX_ROUNDS + 1):
        print(f"[3] Student-Critic round {round_id}/{MAX_ROUNDS}", flush=True)
        critiques, duration = generate_batch(
            [
                critic_messages(questions[i]["problem"], critic_current[i]["text"])
                for i in range(n)
            ],
            rep_seed + 200 + 2 * round_id,
        )
        critic_cumulative_time += duration
        before_correct = critic_states[round_id - 1]["correct"]

        updated, duration = generate_batch(
            [
                student_update_messages(
                    questions[i]["problem"],
                    critic_current[i]["text"],
                    critiques[i]["text"],
                )
                for i in range(n)
            ],
            rep_seed + 201 + 2 * round_id,
        )
        critic_cumulative_time += duration

        critic_cumulative_tokens = [
            critic_cumulative_tokens[i]
            + critiques[i]["total_tokens"]
            + updated[i]["total_tokens"]
            for i in range(n)
        ]
        correct = [
            answers_match(updated[i]["answer"], questions[i]["gold"]) for i in range(n)
        ]
        records = [
            {
                "verdict": parse_verdict(critiques[i]["text"]),
                "correct_before": before_correct[i],
                "correct_after": correct[i],
            }
            for i in range(n)
        ]

        critic_states[round_id] = {
            "outputs": updated,
            "critiques": critiques,
            "correct": correct,
            "tokens": critic_cumulative_tokens.copy(),
            "cumulative_wall_time": critic_cumulative_time,
        }
        critic_round_metrics[round_id] = {
            **critic_metrics(records),
            "corrections_vs_previous_round": correction_counts(before_correct, correct),
        }
        critic_current = updated

    # --------------------------------------------------------
    # Debate: A0 + B0, then synchronous peer updates.
    # A judge is run at every checkpoint so R=0,1,2,3 are comparable.
    # Earlier checkpoint judges are measurement-only and do not affect later rounds.
    # --------------------------------------------------------
    print("[4] Debate independent second solution", flush=True)
    b0, b0_time = generate_batch(
        [solver_messages(q["problem"]) for q in questions],
        rep_seed + 400,
    )
    agent_a = initial
    agent_b = b0
    debate_path_tokens = [
        initial[i]["total_tokens"] + b0[i]["total_tokens"] for i in range(n)
    ]
    debate_path_time = initial_time + b0_time
    debate_states = {}
    debate_judge_overhead_time = 0.0

    for round_id in range(0, MAX_ROUNDS + 1):
        print(f"[5] Debate checkpoint R={round_id}", flush=True)
        judge, judge_time = generate_batch(
            [
                judge_messages(
                    questions[i]["problem"],
                    agent_a[i]["text"],
                    agent_b[i]["text"],
                )
                for i in range(n)
            ],
            rep_seed + 500 + round_id,
        )
        debate_judge_overhead_time += judge_time

        correct = [
            answers_match(judge[i]["answer"], questions[i]["gold"]) for i in range(n)
        ]
        a_correct = [
            answers_match(agent_a[i]["answer"], questions[i]["gold"]) for i in range(n)
        ]
        b_correct = [
            answers_match(agent_b[i]["answer"], questions[i]["gold"]) for i in range(n)
        ]
        disagreement = [
            not answers_match(agent_a[i]["answer"], agent_b[i]["answer"])
            for i in range(n)
        ]
        protocol_tokens = [
            debate_path_tokens[i] + judge[i]["total_tokens"] for i in range(n)
        ]

        debate_states[round_id] = {
            "judge": judge,
            "agent_a": agent_a,
            "agent_b": agent_b,
            "correct": correct,
            "agent_a_correct": a_correct,
            "agent_b_correct": b_correct,
            "disagreement": disagreement,
            "tokens": protocol_tokens,
            "cumulative_wall_time": debate_path_time + judge_time,
        }

        if round_id == MAX_ROUNDS:
            break

        print(f"[6] Debate interaction round {round_id + 1}/{MAX_ROUNDS}", flush=True)
        a_new, duration_a = generate_batch(
            [
                debate_messages(
                    questions[i]["problem"],
                    agent_a[i]["text"],
                    agent_b[i]["text"],
                    "Agent A",
                )
                for i in range(n)
            ],
            rep_seed + 600 + 2 * (round_id + 1),
        )
        b_new, duration_b = generate_batch(
            [
                debate_messages(
                    questions[i]["problem"],
                    agent_b[i]["text"],
                    agent_a[i]["text"],
                    "Agent B",
                )
                for i in range(n)
            ],
            rep_seed + 601 + 2 * (round_id + 1),
        )

        debate_path_time += duration_a + duration_b
        debate_path_tokens = [
            debate_path_tokens[i] + a_new[i]["total_tokens"] + b_new[i]["total_tokens"]
            for i in range(n)
        ]
        agent_a, agent_b = a_new, b_new

    # --------------------------------------------------------
    # Checkpoint summaries
    # --------------------------------------------------------
    checkpoints = []
    for round_id in ROUND_VALUES:
        checkpoints.append(
            summarize_checkpoint(
                "self_refine",
                round_id,
                self_states[round_id]["correct"],
                self_states[round_id]["tokens"],
                1 + 2 * round_id,
                self_states[round_id]["cumulative_wall_time"],
                single_correct,
            )
        )
        checkpoints.append(
            summarize_checkpoint(
                "student_critic",
                round_id,
                critic_states[round_id]["correct"],
                critic_states[round_id]["tokens"],
                1 + 2 * round_id,
                critic_states[round_id]["cumulative_wall_time"],
                single_correct,
            )
        )
        checkpoints.append(
            summarize_checkpoint(
                "debate",
                round_id,
                debate_states[round_id]["correct"],
                debate_states[round_id]["tokens"],
                3 + 2 * round_id,
                debate_states[round_id]["cumulative_wall_time"],
                single_correct,
            )
        )

    for method, states in (
        ("self_refine", self_states),
        ("student_critic", critic_states),
        ("debate", debate_states),
    ):
        previous = None
        for round_id in ROUND_VALUES:
            current = states[round_id]["correct"]
            checkpoint = next(
                x for x in checkpoints if x["method"] == method and x["round"] == round_id
            )
            checkpoint["corrections_vs_previous_round"] = (
                None if previous is None else correction_counts(previous, current)
            )
            previous = current

    for round_id in ROUND_VALUES:
        debate_checkpoint = next(
            x for x in checkpoints if x["method"] == "debate" and x["round"] == round_id
        )
        debate_checkpoint.update(
            {
                "agent_a_accuracy": sum(debate_states[round_id]["agent_a_correct"]) / n,
                "agent_b_accuracy": sum(debate_states[round_id]["agent_b_correct"]) / n,
                "disagreement_rate": sum(debate_states[round_id]["disagreement"]) / n,
                "consensus_rate": 1 - sum(debate_states[round_id]["disagreement"]) / n,
            }
        )

        if round_id > 0:
            critic_checkpoint = next(
                x
                for x in checkpoints
                if x["method"] == "student_critic" and x["round"] == round_id
            )
            critic_checkpoint["critic_quality_this_round"] = critic_round_metrics[round_id]

    # Accuracy gain per extra 1k tokens relative to Single.
    single_total_tokens = sum(single_tokens)
    single_accuracy = sum(single_correct) / n
    for checkpoint in checkpoints:
        extra_tokens = checkpoint["protocol_total_tokens"] - single_total_tokens
        checkpoint["extra_tokens_vs_single"] = extra_tokens
        checkpoint["accuracy_gain_per_1k_extra_tokens"] = (
            checkpoint["accuracy_gain_vs_single"] / (extra_tokens / 1000)
            if extra_tokens > 0
            else None
        )

    method_states = {
        "self_refine": {r: self_states[r] for r in ROUND_VALUES},
        "student_critic": {r: critic_states[r] for r in ROUND_VALUES},
        "debate": {r: debate_states[r] for r in ROUND_VALUES},
    }
    by_difficulty = difficulty_summary(questions, method_states, single_correct)

    # --------------------------------------------------------
    # Per-question traces, enough for later failure analysis
    # --------------------------------------------------------
    rows = []
    for i, question in enumerate(questions):
        row = {
            "repetition": repetition,
            "question_id": question["question_id"],
            "dataset_index": question["dataset_index"],
            "level": question["level"],
            "type": question["type"],
            "gold": question["gold"],
            "single": {
                "output": initial[i],
                "correct": single_correct[i],
            },
            "self_refine": {},
            "student_critic": {},
            "debate": {},
        }

        for round_id in ROUND_VALUES:
            self_state = self_states[round_id]
            critic_state = critic_states[round_id]
            debate_state = debate_states[round_id]

            row["self_refine"][str(round_id)] = {
                "output": self_state["outputs"][i],
                "feedback": self_state.get("feedback", [None] * n)[i]
                if round_id > 0
                else None,
                "correct": self_state["correct"][i],
                "cumulative_tokens": self_state["tokens"][i],
            }
            row["student_critic"][str(round_id)] = {
                "output": critic_state["outputs"][i],
                "critique": critic_state.get("critiques", [None] * n)[i]
                if round_id > 0
                else None,
                "verdict": parse_verdict(critic_state["critiques"][i]["text"])
                if round_id > 0
                else None,
                "correct": critic_state["correct"][i],
                "cumulative_tokens": critic_state["tokens"][i],
            }
            row["debate"][str(round_id)] = {
                "agent_a": debate_state["agent_a"][i],
                "agent_b": debate_state["agent_b"][i],
                "judge": debate_state["judge"][i],
                "correct": debate_state["correct"][i],
                "disagreement": debate_state["disagreement"][i],
                "cumulative_protocol_tokens": debate_state["tokens"][i],
            }

        rows.append(row)

    rep_elapsed = time.perf_counter() - rep_start
    summary = {
        "repetition": repetition,
        "questions": n,
        "round_values": ROUND_VALUES,
        "single_accuracy": single_accuracy,
        "single_total_tokens": single_total_tokens,
        "checkpoints": checkpoints,
        "difficulty_analysis": by_difficulty,
        "measurement_note": (
            "Debate judges are run at every round checkpoint for analysis only. "
            "A checkpoint's protocol cost includes only the debate path up to that round "
            "plus that round's judge; earlier checkpoint judges do not affect later debate states."
        ),
        "executed_generation_count": n * (1 + 2 * MAX_ROUNDS + 2 * MAX_ROUNDS + 1 + 2 * MAX_ROUNDS + (MAX_ROUNDS + 1)),
        "debate_checkpoint_judge_overhead_seconds": debate_judge_overhead_time,
        "repetition_wall_time_seconds": rep_elapsed,
    }

    return rows, summary


# ============================================================
# FINAL AGGREGATION
# ============================================================

def aggregate_summaries(summaries):
    grouped = defaultdict(list)
    for summary in summaries:
        for checkpoint in summary["checkpoints"]:
            grouped[(checkpoint["method"], checkpoint["round"])].append(checkpoint)

    checkpoint_means = []
    for (method, round_id), values in sorted(grouped.items()):
        numeric_fields = [
            "accuracy",
            "accuracy_gain_vs_single",
            "protocol_total_tokens",
            "protocol_mean_tokens_per_question",
            "protocol_wall_time_seconds",
            "extra_tokens_vs_single",
            "accuracy_gain_per_1k_extra_tokens",
        ]
        item = {"method": method, "round": round_id, "repetitions": len(values)}

        for field in numeric_fields:
            xs = [v[field] for v in values if v.get(field) is not None]
            item[field] = {
                "mean": mean(xs) if xs else None,
                "std_population": pstdev(xs) if len(xs) > 1 else 0.0 if xs else None,
                "values": xs,
            }

        checkpoint_means.append(item)

    difficulty_grouped = defaultdict(list)
    for summary in summaries:
        for row in summary["difficulty_analysis"]:
            difficulty_grouped[(row["method"], row["round"], row["level"])].append(row)

    difficulty_means = []
    for (method, round_id, level), values in sorted(difficulty_grouped.items()):
        difficulty_means.append(
            {
                "method": method,
                "round": round_id,
                "level": level,
                "questions_per_repetition": values[0]["questions"],
                "accuracy_mean": mean(v["accuracy"] for v in values),
                "gain_vs_single_mean": mean(v["gain_vs_single"] for v in values),
            }
        )

    return {
        "research_question": (
            "How do additional interaction rounds affect accuracy, corrections, cost, "
            "and failure risk, and does this pattern change with MATH difficulty?"
        ),
        "design": {
            "methods": ["self_refine", "student_critic", "debate"],
            "round_values": ROUND_VALUES,
            "questions": N_QUESTIONS,
            "preserved_experiment_a_questions": BASE_A_QUESTIONS,
            "base_experiment_a_seed": BASE_A_SEED,
            "sampling": "Experiment A sample preserved + level-stratified expansion",
            "repetitions": N_REPETITIONS,
            "dataset": "jeggers/competition_math (original/test)",
            "model": MODEL_PATH,
        },
        "checkpoint_means": checkpoint_means,
        "difficulty_means": difficulty_means,
        "mean_repetition_wall_time_seconds": mean(
            s["repetition_wall_time_seconds"] for s in summaries
        ),
        "total_executed_generations": sum(s["executed_generation_count"] for s in summaries),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    initialize_engine()

    print("=" * 80)
    print("EXPERIMENT 7 / B - ROUND SCALING x MATH DIFFICULTY")
    print("=" * 80)
    print("Model:", MODEL_PATH)
    print("Questions:", N_QUESTIONS)
    print("Preserved Experiment A questions:", BASE_A_QUESTIONS)
    print("Repetitions:", N_REPETITIONS)
    print("Round checkpoints:", ROUND_VALUES)
    print("Results:", RESULTS_DIR)
    print("=" * 80)

    experiment_start = time.perf_counter()
    questions = load_math_sample()
    save_json(RESULTS_DIR / "sampled_questions.json", questions)

    raw_path = RESULTS_DIR / "raw_results.jsonl"
    if raw_path.exists():
        raw_path.unlink()

    summaries = []
    for repetition in range(1, N_REPETITIONS + 1):
        rows, summary = run_repetition(questions, repetition)
        append_jsonl(raw_path, rows)
        save_json(RESULTS_DIR / f"repetition_{repetition}_summary.json", summary)
        summaries.append(summary)
        save_json(RESULTS_DIR / "summaries.json", summaries)

        print(f"\nRepetition {repetition} completed in {summary['repetition_wall_time_seconds']:.1f}s")
        for checkpoint in summary["checkpoints"]:
            print(
                f"  {checkpoint['method']:<16} R={checkpoint['round']} "
                f"acc={checkpoint['accuracy']:.2%} "
                f"tokens={checkpoint['protocol_total_tokens']:,}"
            )

    final_summary = aggregate_summaries(summaries)
    final_summary["total_experiment_wall_time_seconds"] = time.perf_counter() - experiment_start
    save_json(RESULTS_DIR / "final_summary.json", final_summary)

    print("\n" + "=" * 80)
    print("FINAL MEAN CHECKPOINT RESULTS")
    print("=" * 80)
    for item in final_summary["checkpoint_means"]:
        print(
            f"{item['method']:<16} R={item['round']} "
            f"acc={item['accuracy']['mean']:.2%} "
            f"gain={item['accuracy_gain_vs_single']['mean']:+.2%}"
        )

    print(f"\nTotal wall time: {final_summary['total_experiment_wall_time_seconds']:.1f}s")
    print(f"Total executed generations: {final_summary['total_executed_generations']:,}")
    print("Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
