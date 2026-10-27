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
    "RESULTS_SUBDIR",
    "experiment_6_single_forest_selfrefine_studentcritic_debate",
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "200"))
N_REPETITIONS = int(os.environ.get("N_REPETITIONS", "3"))
FOREST_N = int(os.environ.get("FOREST_N", "5"))
REFINE_ROUNDS = int(os.environ.get("REFINE_ROUNDS", "2"))
CRITIC_ROUNDS = int(os.environ.get("CRITIC_ROUNDS", "2"))
DEBATE_ROUNDS = int(os.environ.get("DEBATE_ROUNDS", "1"))

TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "64"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))

DATASET_SEED = int(os.environ.get("DATASET_SEED", "42"))
GENERATION_SEED = int(os.environ.get("GENERATION_SEED", "42"))

if 1 + 2 * REFINE_ROUNDS != FOREST_N:
    raise ValueError("Need 1 + 2*REFINE_ROUNDS == FOREST_N")
if 1 + 2 * CRITIC_ROUNDS != FOREST_N:
    raise ValueError("Need 1 + 2*CRITIC_ROUNDS == FOREST_N")
if 2 + 2 * DEBATE_ROUNDS + 1 != FOREST_N:
    raise ValueError("Need 2 + 2*DEBATE_ROUNDS + 1 == FOREST_N")


tokenizer = None
llm = None

FINAL_RE = re.compile(r"FINAL:\s*(.+?)(?:\n|$)", re.I)
VERDICT_RE = re.compile(r"VERDICT:\s*(ACCEPT|REJECT)", re.I)


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
Check arithmetic, algebra, logic, interpretation and missing cases.
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
Read the other agent's solution, compare it with your own current solution, keep correct reasoning, and revise only if the other agent exposes a concrete error or stronger argument.
End with exactly: FINAL: <answer>
""".strip()

JUDGE_SYSTEM = """
You are the final judge in a mathematical debate.
Given the problem and the final solutions from two agents, decide the best mathematical answer by checking both independently.
End with exactly: FINAL: <answer>
""".strip()


# ============================================================
# FILE HELPERS
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


# ============================================================
# DATASET AND EVALUATION
# ============================================================

def load_math_sample():
    ds = load_dataset("jeggers/competition_math", "original", split="test")
    if N_QUESTIONS > len(ds):
        raise ValueError(f"N_QUESTIONS={N_QUESTIONS} > dataset={len(ds)}")

    rng = random.Random(DATASET_SEED)
    idxs = rng.sample(range(len(ds)), N_QUESTIONS)

    return [
        {
            "question_id": i + 1,
            "dataset_index": idx,
            "problem": ds[idx]["problem"],
            "gold": str(ds[idx]["extracted_solution"]),
            "level": ds[idx].get("level"),
            "type": ds[idx].get("type"),
        }
        for i, idx in enumerate(idxs)
    ]


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


def answers_match(a, b):
    if a is None:
        return False

    try:
        parsed_a = parse_math(a)
        parsed_b = parse_math(b)
        return bool(parsed_a and parsed_b and math_verify(parsed_b, parsed_a))
    except Exception:
        return False


# ============================================================
# MESSAGE BUILDERS
# ============================================================

def build_messages(system, user):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def solver_messages(problem):
    return build_messages(SOLVER_SYSTEM, problem)


def self_feedback_messages(problem, previous_solution):
    prompt = f"""ORIGINAL PROBLEM
{problem}

PREVIOUS SOLUTION
{previous_solution}"""
    return build_messages(SELF_FEEDBACK_SYSTEM, prompt)


def self_refine_messages(problem, previous_solution, feedback):
    prompt = f"""ORIGINAL PROBLEM
{problem}

PREVIOUS SOLUTION
{previous_solution}

SELF-FEEDBACK
{feedback}"""
    return build_messages(SELF_REFINE_SYSTEM, prompt)


def critic_messages(problem, previous_solution):
    prompt = f"""ORIGINAL PROBLEM
{problem}

STUDENT SOLUTION
{previous_solution}"""
    return build_messages(CRITIC_SYSTEM, prompt)


def student_update_messages(problem, previous_solution, critique):
    prompt = f"""ORIGINAL PROBLEM
{problem}

CURRENT STUDENT SOLUTION
{previous_solution}

CRITIC FEEDBACK
{critique}"""
    return build_messages(STUDENT_UPDATE_SYSTEM, prompt)


def debate_messages(problem, own_solution, other_solution, role):
    prompt = f"""ORIGINAL PROBLEM
{problem}

YOUR CURRENT SOLUTION ({role})
{own_solution}

OTHER AGENT SOLUTION
{other_solution}"""
    return build_messages(DEBATE_SYSTEM, prompt)


def judge_messages(problem, agent_a, agent_b):
    prompt = f"""ORIGINAL PROBLEM
{problem}

AGENT A FINAL SOLUTION
{agent_a}

AGENT B FINAL SOLUTION
{agent_b}"""
    return build_messages(JUDGE_SYSTEM, prompt)


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


def generate_batch(messages_batch, seed, n=1):
    prompts = [render_messages(messages) for messages in messages_batch]
    params = SamplingParams(
        n=n,
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
        candidates = []

        for candidate in request_output.outputs:
            completion_tokens = len(candidate.token_ids)
            candidates.append(
                {
                    "text": candidate.text,
                    "answer": extract_final_text(candidate.text),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "cumulative_logprob": getattr(
                        candidate, "cumulative_logprob", None
                    ),
                }
            )

        parsed.append(candidates)

    return parsed, wall_time


# ============================================================
# VOTING AND METRICS
# ============================================================

def semantic_groups(candidates):
    groups = []

    for candidate in [x for x in candidates if x["answer"] is not None]:
        placed = False

        for group in groups:
            if answers_match(candidate["answer"], group[0]["answer"]):
                group.append(candidate)
                placed = True
                break

        if not placed:
            groups.append([candidate])

    return groups


def semantic_vote(candidates, seed):
    groups = semantic_groups(candidates)
    if not groups:
        return None, True, 0, 0

    max_votes = max(len(group) for group in groups)
    winners = [group for group in groups if len(group) == max_votes]
    tie = len(winners) > 1
    winner = random.Random(seed).choice(winners)[0]["answer"]

    return winner, tie, len(groups), max_votes


def correction_flags(initial_correct, final_correct):
    return {
        "positive_correction": not initial_correct and final_correct,
        "negative_correction": initial_correct and not final_correct,
    }


def accuracy(rows, field):
    return sum(bool(row[field]) for row in rows) / len(rows)


def critic_metrics(records):
    true_reject = false_reject = true_accept = false_accept = unknown = 0

    for record in records:
        verdict = record["verdict"]
        correct = record["student_correct_before"]

        if verdict == "UNKNOWN":
            unknown += 1
        elif verdict == "REJECT" and not correct:
            true_reject += 1
        elif verdict == "REJECT" and correct:
            false_reject += 1
        elif verdict == "ACCEPT" and correct:
            true_accept += 1
        elif verdict == "ACCEPT" and not correct:
            false_accept += 1

    precision = (
        true_reject / (true_reject + false_reject)
        if true_reject + false_reject
        else None
    )
    recall = (
        true_reject / (true_reject + false_accept)
        if true_reject + false_accept
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )

    return {
        "true_reject": true_reject,
        "false_reject": false_reject,
        "true_accept": true_accept,
        "false_accept": false_accept,
        "unknown_verdict": unknown,
        "error_detection_precision": precision,
        "error_detection_recall": recall,
        "error_detection_f1": f1,
    }


# ============================================================
# ONE REPETITION
# ============================================================

def run_repetition(questions, repetition):
    rep_seed = GENERATION_SEED + 10000 * repetition

    # Shared initial solution / Single
    initial_outputs, single_time = generate_batch(
        [solver_messages(q["problem"]) for q in questions],
        rep_seed + 1,
    )
    initial = [output[0] for output in initial_outputs]

    # --------------------------------------------------------
    # Agent Forest
    # --------------------------------------------------------
    forest_extra, forest_time = generate_batch(
        [solver_messages(q["problem"]) for q in questions],
        rep_seed + 2,
        n=FOREST_N - 1,
    )

    forest = []
    for i in range(len(questions)):
        candidates = [initial[i]] + forest_extra[i]
        answer, tie, num_groups, max_votes = semantic_vote(
            candidates,
            rep_seed + 3000 + i,
        )
        forest.append(
            {
                "candidates": candidates,
                "final_answer": answer,
                "vote_tie": tie,
                "num_groups": num_groups,
                "max_votes": max_votes,
            }
        )

    # --------------------------------------------------------
    # Self-Refine
    # --------------------------------------------------------
    self_current = [item["text"] for item in initial]
    self_rounds = [[] for _ in questions]
    self_round_accuracy = []
    self_time = 0.0

    for round_id in range(1, REFINE_ROUNDS + 1):
        feedback_outputs, duration = generate_batch(
            [
                self_feedback_messages(questions[i]["problem"], self_current[i])
                for i in range(len(questions))
            ],
            rep_seed + 100 + 2 * round_id,
        )
        self_time += duration
        feedback = [output[0] for output in feedback_outputs]

        refined_outputs, duration = generate_batch(
            [
                self_refine_messages(
                    questions[i]["problem"],
                    self_current[i],
                    feedback[i]["text"],
                )
                for i in range(len(questions))
            ],
            rep_seed + 101 + 2 * round_id,
        )
        self_time += duration
        refined = [output[0] for output in refined_outputs]

        round_accuracy = sum(
            answers_match(refined[i]["answer"], questions[i]["gold"])
            for i in range(len(questions))
        ) / len(questions)
        self_round_accuracy.append(round_accuracy)

        for i in range(len(questions)):
            self_rounds[i].append(
                {
                    "round": round_id,
                    "feedback": feedback[i],
                    "refined": refined[i],
                }
            )

        self_current = [item["text"] for item in refined]

    self_final = [rounds[-1]["refined"] for rounds in self_rounds]

    # --------------------------------------------------------
    # Student-Critic
    # --------------------------------------------------------
    critic_current = [item["text"] for item in initial]
    critic_rounds = [[] for _ in questions]
    critic_records = []
    critic_round_stats = []
    critic_time = 0.0

    for round_id in range(1, CRITIC_ROUNDS + 1):
        critique_outputs, duration = generate_batch(
            [
                critic_messages(questions[i]["problem"], critic_current[i])
                for i in range(len(questions))
            ],
            rep_seed + 200 + 2 * round_id,
        )
        critic_time += duration
        critiques = [output[0] for output in critique_outputs]

        before_answers = [extract_final_text(text) for text in critic_current]
        before_correct = [
            answers_match(before_answers[i], questions[i]["gold"])
            for i in range(len(questions))
        ]

        update_outputs, duration = generate_batch(
            [
                student_update_messages(
                    questions[i]["problem"],
                    critic_current[i],
                    critiques[i]["text"],
                )
                for i in range(len(questions))
            ],
            rep_seed + 201 + 2 * round_id,
        )
        critic_time += duration
        updated = [output[0] for output in update_outputs]

        round_records = []
        for i in range(len(questions)):
            verdict = parse_verdict(critiques[i]["text"])
            after_correct = answers_match(updated[i]["answer"], questions[i]["gold"])

            record = {
                "round": round_id,
                "verdict": verdict,
                "critique": critiques[i],
                "student_before_answer": before_answers[i],
                "student_correct_before": before_correct[i],
                "student_after": updated[i],
                "student_correct_after": after_correct,
            }
            critic_rounds[i].append(record)

            compact_record = {
                "round": round_id,
                "verdict": verdict,
                "student_correct_before": before_correct[i],
                "student_correct_after": after_correct,
            }
            critic_records.append(compact_record)
            round_records.append(compact_record)

        critic_round_stats.append(
            {
                "round": round_id,
                "student_accuracy_after": sum(
                    item["student_correct_after"] for item in round_records
                )
                / len(round_records),
                **critic_metrics(round_records),
            }
        )

        critic_current = [item["text"] for item in updated]

    critic_final = [rounds[-1]["student_after"] for rounds in critic_rounds]

    # --------------------------------------------------------
    # Multi-Agent Debate
    # --------------------------------------------------------
    b0_outputs, debate_time = generate_batch(
        [solver_messages(q["problem"]) for q in questions],
        rep_seed + 400,
    )

    agent_a_current = [item["text"] for item in initial]
    agent_b_current = [item[0]["text"] for item in b0_outputs]
    debate_rounds = [[] for _ in questions]

    initial_disagreement = [
        not answers_match(
            extract_final_text(agent_a_current[i]),
            extract_final_text(agent_b_current[i]),
        )
        for i in range(len(questions))
    ]

    for round_id in range(1, DEBATE_ROUNDS + 1):
        a_outputs, duration = generate_batch(
            [
                debate_messages(
                    questions[i]["problem"],
                    agent_a_current[i],
                    agent_b_current[i],
                    "Agent A",
                )
                for i in range(len(questions))
            ],
            rep_seed + 410 + 2 * round_id,
        )
        debate_time += duration

        b_outputs, duration = generate_batch(
            [
                debate_messages(
                    questions[i]["problem"],
                    agent_b_current[i],
                    agent_a_current[i],
                    "Agent B",
                )
                for i in range(len(questions))
            ],
            rep_seed + 411 + 2 * round_id,
        )
        debate_time += duration

        agent_a_new = [item[0] for item in a_outputs]
        agent_b_new = [item[0] for item in b_outputs]

        for i in range(len(questions)):
            debate_rounds[i].append(
                {
                    "round": round_id,
                    "agent_a": agent_a_new[i],
                    "agent_b": agent_b_new[i],
                }
            )

        agent_a_current = [item["text"] for item in agent_a_new]
        agent_b_current = [item["text"] for item in agent_b_new]

    judge_outputs, duration = generate_batch(
        [
            judge_messages(
                questions[i]["problem"],
                agent_a_current[i],
                agent_b_current[i],
            )
            for i in range(len(questions))
        ],
        rep_seed + 499,
    )
    debate_time += duration
    judge = [output[0] for output in judge_outputs]

    # --------------------------------------------------------
    # Per-question results
    # --------------------------------------------------------
    rows = []

    for i, question in enumerate(questions):
        single_ok = answers_match(initial[i]["answer"], question["gold"])
        forest_ok = answers_match(forest[i]["final_answer"], question["gold"])
        self_ok = answers_match(self_final[i]["answer"], question["gold"])
        critic_ok = answers_match(critic_final[i]["answer"], question["gold"])
        debate_ok = answers_match(judge[i]["answer"], question["gold"])

        final_disagreement = not answers_match(
            extract_final_text(agent_a_current[i]),
            extract_final_text(agent_b_current[i]),
        )

        forest_correction = correction_flags(single_ok, forest_ok)
        self_correction = correction_flags(single_ok, self_ok)
        critic_correction = correction_flags(single_ok, critic_ok)
        debate_correction = correction_flags(single_ok, debate_ok)

        forest_tokens = sum(item["total_tokens"] for item in forest[i]["candidates"])
        self_tokens = initial[i]["total_tokens"] + sum(
            item["feedback"]["total_tokens"] + item["refined"]["total_tokens"]
            for item in self_rounds[i]
        )
        critic_tokens = initial[i]["total_tokens"] + sum(
            item["critique"]["total_tokens"]
            + item["student_after"]["total_tokens"]
            for item in critic_rounds[i]
        )
        debate_tokens = (
            initial[i]["total_tokens"]
            + b0_outputs[i][0]["total_tokens"]
            + sum(
                item["agent_a"]["total_tokens"] + item["agent_b"]["total_tokens"]
                for item in debate_rounds[i]
            )
            + judge[i]["total_tokens"]
        )

        rows.append(
            {
                "repetition": repetition,
                "question_id": question["question_id"],
                "dataset_index": question["dataset_index"],
                "level": question["level"],
                "type": question["type"],
                "gold": question["gold"],
                "single_answer": initial[i]["answer"],
                "single_correct": single_ok,
                "single_tokens": initial[i]["total_tokens"],
                "forest_answer": forest[i]["final_answer"],
                "forest_correct": forest_ok,
                "forest_positive_correction": forest_correction["positive_correction"],
                "forest_negative_correction": forest_correction["negative_correction"],
                "forest_vote_tie": forest[i]["vote_tie"],
                "forest_num_groups": forest[i]["num_groups"],
                "forest_max_votes": forest[i]["max_votes"],
                "forest_tokens": forest_tokens,
                "self_refine_answer": self_final[i]["answer"],
                "self_refine_correct": self_ok,
                "self_refine_positive_correction": self_correction[
                    "positive_correction"
                ],
                "self_refine_negative_correction": self_correction[
                    "negative_correction"
                ],
                "self_refine_tokens": self_tokens,
                "student_critic_answer": critic_final[i]["answer"],
                "student_critic_correct": critic_ok,
                "student_critic_positive_correction": critic_correction[
                    "positive_correction"
                ],
                "student_critic_negative_correction": critic_correction[
                    "negative_correction"
                ],
                "student_critic_tokens": critic_tokens,
                "debate_answer": judge[i]["answer"],
                "debate_correct": debate_ok,
                "debate_positive_correction": debate_correction["positive_correction"],
                "debate_negative_correction": debate_correction["negative_correction"],
                "debate_initial_disagreement": initial_disagreement[i],
                "debate_final_disagreement": final_disagreement,
                "debate_reached_consensus": initial_disagreement[i]
                and not final_disagreement,
                "debate_tokens": debate_tokens,
                "single_trace": initial[i],
                "forest_trace": forest[i],
                "self_refine_trace": self_rounds[i],
                "student_critic_trace": critic_rounds[i],
                "debate_trace": {
                    "agent_b_initial": b0_outputs[i][0],
                    "rounds": debate_rounds[i],
                    "judge": judge[i],
                },
            }
        )

    # --------------------------------------------------------
    # Repetition summary
    # --------------------------------------------------------
    def correction_summary(prefix):
        positive = sum(row[f"{prefix}_positive_correction"] for row in rows)
        negative = sum(row[f"{prefix}_negative_correction"] for row in rows)
        baseline_wrong = sum(not row["single_correct"] for row in rows)
        baseline_right = sum(row["single_correct"] for row in rows)

        return {
            "positive_corrections": positive,
            "negative_corrections": negative,
            "positive_correction_rate": (
                positive / baseline_wrong if baseline_wrong else None
            ),
            "negative_correction_rate": (
                negative / baseline_right if baseline_right else None
            ),
        }

    summary = {
        "repetition": repetition,
        "questions": len(rows),
        "single_accuracy": accuracy(rows, "single_correct"),
        "forest_accuracy": accuracy(rows, "forest_correct"),
        "self_refine_accuracy": accuracy(rows, "self_refine_correct"),
        "student_critic_accuracy": accuracy(rows, "student_critic_correct"),
        "debate_accuracy": accuracy(rows, "debate_correct"),
        "forest_corrections": correction_summary("forest"),
        "self_refine_corrections": correction_summary("self_refine"),
        "student_critic_corrections": correction_summary("student_critic"),
        "debate_corrections": correction_summary("debate"),
        "self_refine_round_accuracy": self_round_accuracy,
        "student_critic_round_stats": critic_round_stats,
        "critic_quality": critic_metrics(critic_records),
        "debate_initial_disagreement_rate": sum(
            row["debate_initial_disagreement"] for row in rows
        )
        / len(rows),
        "debate_final_disagreement_rate": sum(
            row["debate_final_disagreement"] for row in rows
        )
        / len(rows),
        "debate_consensus_rate": sum(
            row["debate_reached_consensus"] for row in rows
        )
        / len(rows),
        "single_total_tokens": sum(row["single_tokens"] for row in rows),
        "forest_total_tokens": sum(row["forest_tokens"] for row in rows),
        "self_refine_total_tokens": sum(row["self_refine_tokens"] for row in rows),
        "student_critic_total_tokens": sum(
            row["student_critic_tokens"] for row in rows
        ),
        "debate_total_tokens": sum(row["debate_tokens"] for row in rows),
        "stage_wall_time_seconds": {
            "single": single_time,
            "forest_extra": forest_time,
            "self_refine_extra": self_time,
            "student_critic_extra": critic_time,
            "debate_total": debate_time,
        },
    }

    return rows, summary


# ============================================================
# MAIN
# ============================================================

def main():
    experiment_start = time.perf_counter()

    initialize_engine()
    questions = load_math_sample()
    save_json(RESULTS_DIR / "sampled_questions.json", questions)

    raw_results_path = RESULTS_DIR / "raw_results.jsonl"
    if raw_results_path.exists():
        raw_results_path.unlink()

    summaries = []

    for repetition in range(1, N_REPETITIONS + 1):
        repetition_start = time.perf_counter()
        rows, summary = run_repetition(questions, repetition)
        summary["repetition_wall_time_seconds"] = (
            time.perf_counter() - repetition_start
        )

        append_jsonl(raw_results_path, rows)
        save_json(
            RESULTS_DIR / f"repetition_{repetition}_summary.json",
            summary,
        )

        summaries.append(summary)
        save_json(RESULTS_DIR / "summaries.json", summaries)

        print(
            f"Rep {repetition}: "
            f"single={summary['single_accuracy']:.3f} "
            f"forest={summary['forest_accuracy']:.3f} "
            f"self={summary['self_refine_accuracy']:.3f} "
            f"critic={summary['student_critic_accuracy']:.3f} "
            f"debate={summary['debate_accuracy']:.3f} "
            f"time={summary['repetition_wall_time_seconds']:.1f}s",
            flush=True,
        )

    fields = [
        "single_accuracy",
        "forest_accuracy",
        "self_refine_accuracy",
        "student_critic_accuracy",
        "debate_accuracy",
    ]

    final_summary = {}
    for field in fields:
        values = [summary[field] for summary in summaries]
        final_summary[field] = {
            "values": values,
            "mean": mean(values),
            "std_population": pstdev(values) if len(values) > 1 else 0.0,
        }

    final_summary["total_experiment_wall_time_seconds"] = (
        time.perf_counter() - experiment_start
    )
    final_summary["mean_repetition_wall_time_seconds"] = mean(
        summary["repetition_wall_time_seconds"] for summary in summaries
    )

    save_json(RESULTS_DIR / "final_summary.json", final_summary)
    print(json.dumps(final_summary, indent=2), flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
