import os
import re
import csv
import json
import time
import random
from collections import Counter
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIGURATION — NARVAL / LOCAL QWEN
# ============================================================

PROJECT_ROOT = "/home/isa1909/projects/def-sigmagao/isa1909"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-7B-Instruct")
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", "competition_math_test")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiment_4_role_specialization_qwen25_7b")
MODEL_NAME = "Qwen2.5-7B-Instruct"

N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "2"))
N_REPETITIONS = int(os.environ.get("N_REPETITIONS", "1"))
DATASET_SEED = int(os.environ.get("DATASET_SEED", "42"))
GENERATION_SEED = int(os.environ.get("GENERATION_SEED", "42"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "8192"))

TOKENIZER = None
LOCAL_MODEL = None

def load_local_model():
    global TOKENIZER, LOCAL_MODEL
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Local model directory not found: {MODEL_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Submit this program as a GPU Slurm job.")

    print("=" * 76, flush=True)
    print("LOADING LOCAL MODEL", flush=True)
    print("=" * 76, flush=True)
    print("Model path:", MODEL_PATH, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("GPU memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), flush=True)

    TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    LOCAL_MODEL.to("cuda")
    LOCAL_MODEL.eval()

    torch.manual_seed(GENERATION_SEED)
    torch.cuda.manual_seed_all(GENERATION_SEED)
    print("Model loaded successfully.", flush=True)
    print()


# ============================================================
# NORMALIZATION
# ============================================================

def strip_latex_wrappers(value):
    if value is None:
        return None

    value = str(value).strip()

    value = value.replace("\\left", "")
    value = value.replace("\\right", "")
    value = value.replace("\\!", "")
    value = value.replace("\\,", "")
    value = value.replace("\\;", "")
    value = value.replace("\\ ", "")

    return value.strip()


def normalize_math_answer(value):
    if value is None:
        return None

    value = strip_latex_wrappers(value)

    if value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()

    boxed_match = re.fullmatch(
        r"\\boxed\{(.+)\}",
        value,
    )

    if boxed_match:
        value = boxed_match.group(1).strip()

    value = re.sub(r"\s+", "", value)

    value = value.replace("\\dfrac", "\\frac")
    value = value.replace("\\tfrac", "\\frac")

    if re.fullmatch(
        r"-?\d{1,3}(,\d{3})+",
        value,
    ):
        value = value.replace(",", "")

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return f"{number:.10f}".rstrip("0").rstrip(".")

    except ValueError:
        pass

    return value.lower()


def extract_final_answer(text):
    if text is None:
        return None

    # Preferred explicit final answer
    match = re.search(
        r"FINAL:\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return normalize_math_answer(
            match.group(1)
        )

    # Fallback for simple boxed answers
    matches = re.findall(
        r"\\boxed\{([^{}]+)\}",
        text,
    )

    if matches:
        return normalize_math_answer(
            matches[-1]
        )

    return None


def answers_match(prediction, ground_truth):
    if prediction is None:
        return False

    return (
        normalize_math_answer(prediction)
        ==
        normalize_math_answer(ground_truth)
    )


# ============================================================
# DATASET
# ============================================================

def load_math_sample():
    print("Loading MATH dataset from local storage...", flush=True)
    print("Dataset path:", DATASET_PATH, flush=True)

    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"Local dataset directory not found: {DATASET_PATH}")

    dataset = load_from_disk(DATASET_PATH)
    rng = random.Random(DATASET_SEED)

    if N_QUESTIONS > len(dataset):
        raise ValueError(f"N_QUESTIONS={N_QUESTIONS} but dataset has only {len(dataset)} rows.")

    indices = rng.sample(range(len(dataset)), N_QUESTIONS)
    questions = []
    for local_id, dataset_index in enumerate(indices, start=1):
        row = dataset[dataset_index]
        ground_truth = row["extracted_solution"]
        questions.append({
            "question_id": local_id,
            "dataset_index": dataset_index,
            "question": row["problem"],
            "ground_truth_raw": ground_truth,
            "ground_truth": normalize_math_answer(ground_truth),
            "level": row.get("level"),
            "type": row.get("type"),
        })

    print(f"Loaded {len(questions)} questions.", flush=True)
    return questions


# ============================================================
# LOCAL MODEL INFERENCE
# ============================================================
# ============================================================

def call_model(
    system_prompt,
    user_prompt,
):
    if TOKENIZER is None or LOCAL_MODEL is None:
        raise RuntimeError("Local model has not been loaded.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = TOKENIZER(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    prompt_tokens = int(input_ids.shape[1])

    start = time.perf_counter()
    with torch.inference_mode():
        generated = LOCAL_MODEL.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.95,
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
            use_cache=True,
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    new_tokens = generated[0, prompt_tokens:]
    completion_tokens = int(new_tokens.shape[0])
    text = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "text": text,
        "answer": extract_final_answer(text),
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# ============================================================
# COMMON SOLVER
# ============================================================

def call_solver(question):
    system_prompt = """
You are a mathematical problem-solving assistant.

Solve the competition mathematics problem carefully and
independently.

Reason step by step.

At the end, provide the final answer on a separate line
exactly in the format:

FINAL: <answer>
""".strip()

    return call_model(
        system_prompt,
        question,
    )


# ============================================================
# CONDITION 1 — SINGLE WRITER
# ============================================================

def run_single(question):
    start = time.perf_counter()

    result = call_solver(question)

    wall_time = (
        time.perf_counter() - start
    )

    return {
        "final_answer": result["answer"],
        "response": result,
        "total_tokens":
            result["total_tokens"],
        "wall_time": wall_time,
    }


# ============================================================
# CONDITION 2 — AGENT FOREST N=3
# ============================================================

def majority_vote(answers):
    valid = [
        answer
        for answer in answers
        if answer is not None
    ]

    if not valid:
        return None

    counts = Counter(valid)

    max_votes = max(
        counts.values()
    )

    winners = [
        answer
        for answer, count
        in counts.items()
        if count == max_votes
    ]

    return sorted(winners)[0]


def run_agent_forest_3(question):
    # Three independent stochastic generations using one local GPU model.
    # Sequential execution avoids three Python threads competing for the same GPU.
    start = time.perf_counter()
    results = []

    for agent_id in range(1, 4):
        try:
            results.append(call_solver(question))
        except Exception as error:
            print(f"Agent Forest agent {agent_id} error:", repr(error), flush=True)

    wall_time = time.perf_counter() - start
    answers = [result["answer"] for result in results]
    final_answer = majority_vote(answers)
    tokens = [
        result["total_tokens"]
        for result in results
        if result["total_tokens"] is not None
    ]

    return {
        "answers": answers,
        "responses": results,
        "final_answer": final_answer,
        "total_tokens": sum(tokens) if tokens else None,
        "wall_time": wall_time,
        "disagreement": len(set(x for x in answers if x is not None)) > 1,
    }


# ============================================================
# CONDITION 3 — SELF-REFLECTION
# ============================================================

def run_self_reflection(question):
    start = time.perf_counter()

    # --------------------------------------------------------
    # Call 1 — Draft
    # --------------------------------------------------------

    draft_system = """
You are solving a competition mathematics problem.

Produce a careful draft solution.

Reason step by step and provide a proposed final answer.

At the end write:

FINAL: <answer>
""".strip()

    draft = call_model(
        draft_system,
        question,
    )

    # --------------------------------------------------------
    # Call 2 — Critique
    # --------------------------------------------------------

    critique_system = """
You are critically reviewing your own previous mathematical
solution.

Inspect it carefully for:
- arithmetic errors;
- algebraic errors;
- logical gaps;
- missing cases;
- incorrect assumptions;
- incorrect final answer.

Do not simply agree with the previous solution.

Explain any error you find and state what should be fixed.

At the end provide your current recommended answer as:

FINAL: <answer>
""".strip()

    critique_prompt = f"""
ORIGINAL PROBLEM

{question}

==================================================

PREVIOUS DRAFT

{draft["text"]}
""".strip()

    critique = call_model(
        critique_system,
        critique_prompt,
    )

    # --------------------------------------------------------
    # Call 3 — Revision
    # --------------------------------------------------------

    revision_system = """
You are revising your mathematical solution after receiving
a critique.

Use the original problem, the draft, and the critique to
produce the best final solution.

Check the mathematics independently before accepting the
critique.

At the end write exactly:

FINAL: <answer>
""".strip()

    revision_prompt = f"""
ORIGINAL PROBLEM

{question}

==================================================

DRAFT

{draft["text"]}

==================================================

CRITIQUE

{critique["text"]}
""".strip()

    revision = call_model(
        revision_system,
        revision_prompt,
    )

    wall_time = (
        time.perf_counter() - start
    )

    token_values = [
        draft["total_tokens"],
        critique["total_tokens"],
        revision["total_tokens"],
    ]

    valid_tokens = [
        value
        for value in token_values
        if value is not None
    ]

    return {
        "draft_answer":
            draft["answer"],
        "critique_answer":
            critique["answer"],
        "final_answer":
            revision["answer"],

        "draft":
            draft,
        "critique":
            critique,
        "revision":
            revision,

        "total_tokens":
            sum(valid_tokens)
            if valid_tokens
            else None,

        "wall_time":
            wall_time,

        "critique_changed_answer":
            critique["answer"]
            != draft["answer"],

        "revision_changed_answer":
            revision["answer"]
            != draft["answer"],
    }


# ============================================================
# CONDITION 4 — PEER REVIEW
# ============================================================

def run_peer_review(question):
    start = time.perf_counter()

    # --------------------------------------------------------
    # Writer
    # --------------------------------------------------------

    writer_system = """
You are the Writer in a mathematical peer-review process.

Produce a complete proposed solution to the competition
mathematics problem.

Reason carefully and independently.

At the end provide:

FINAL: <answer>
""".strip()

    writer = call_model(
        writer_system,
        question,
    )

    # --------------------------------------------------------
    # Reviewer
    # --------------------------------------------------------

    reviewer_system = """
You are an independent Reviewer in a mathematical peer-review
process.

You receive the original problem and a proposed solution
written by another agent.

Critically inspect the solution for:
- arithmetic mistakes;
- algebraic mistakes;
- logical gaps;
- incorrect assumptions;
- missing cases;
- incorrect interpretation;
- incorrect final answer.

Do not accept the Writer's answer merely because it looks
plausible.

Explain your review and provide your recommended answer as:

FINAL: <answer>
""".strip()

    reviewer_prompt = f"""
ORIGINAL PROBLEM

{question}

==================================================

WRITER'S SOLUTION

{writer["text"]}
""".strip()

    reviewer = call_model(
        reviewer_system,
        reviewer_prompt,
    )

    # --------------------------------------------------------
    # Associate Editor
    # --------------------------------------------------------

    editor_system = """
You are the Associate Editor in a mathematical peer-review
process.

You receive:
1. the original problem;
2. the Writer's solution;
3. the independent Reviewer's report.

Make the final decision.

Check both arguments independently. The Writer may be wrong,
but the Reviewer may also be wrong.

Resolve disagreements using your own mathematical reasoning.

At the end provide the final answer exactly as:

FINAL: <answer>
""".strip()

    editor_prompt = f"""
ORIGINAL PROBLEM

{question}

==================================================

WRITER

{writer["text"]}

==================================================

REVIEWER

{reviewer["text"]}
""".strip()

    editor = call_model(
        editor_system,
        editor_prompt,
    )

    wall_time = (
        time.perf_counter() - start
    )

    token_values = [
        writer["total_tokens"],
        reviewer["total_tokens"],
        editor["total_tokens"],
    ]

    valid_tokens = [
        value
        for value in token_values
        if value is not None
    ]

    return {
        "writer_answer":
            writer["answer"],
        "reviewer_answer":
            reviewer["answer"],
        "final_answer":
            editor["answer"],

        "writer":
            writer,
        "reviewer":
            reviewer,
        "editor":
            editor,

        "total_tokens":
            sum(valid_tokens)
            if valid_tokens
            else None,

        "wall_time":
            wall_time,

        "reviewer_changed_writer":
            reviewer["answer"]
            != writer["answer"],

        "editor_changed_writer":
            editor["answer"]
            != writer["answer"],

        "editor_agreed_with_reviewer":
            editor["answer"]
            == reviewer["answer"],
    }


# ============================================================
# SAVE UTILITIES
# ============================================================

def save_json(path, data):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_csv(path, rows):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 76)
    print("NARVAL LOCAL-MODEL ENVIRONMENT")
    print("=" * 76)
    print("Model:", MODEL_NAME)
    print("Model path:", MODEL_PATH)
    print("Dataset path:", DATASET_PATH)
    print("Results directory:", RESULTS_DIR)
    print()

    load_local_model()
    questions = load_math_sample()

    save_json(
        os.path.join(
            RESULTS_DIR,
            "sampled_questions.json",
        ),
        questions,
    )

    all_rows = []
    summaries = []

    print()
    print("=" * 76)
    print("EXPERIMENT 4 — ROLE SPECIALIZATION ABLATION")
    print("=" * 76)
    print("Model:", MODEL_NAME)
    print("Questions:", N_QUESTIONS)
    print("Repetitions:", N_REPETITIONS)
    print()
    print("Conditions:")
    print("  Single Writer       = 1 call")
    print("  Agent Forest N=3   = 3 independent calls")
    print("  Self-Reflection    = Draft -> Critique -> Revision")
    print("  Peer Review        = Writer -> Reviewer -> Editor")
    print()

    for repetition in range(
        1,
        N_REPETITIONS + 1,
    ):
        print()
        print("#" * 76)
        print(
            f"REPETITION {repetition}/{N_REPETITIONS}"
        )
        print("#" * 76)

        repetition_rows = []

        for index, item in enumerate(
            questions,
            start=1,
        ):
            question = item["question"]
            truth = item["ground_truth"]

            print()
            print(
                f"[{index:03d}/{N_QUESTIONS}] "
                f"{item['level']} | "
                f"{item['type']}",
                flush=True,
            )

            # =================================================
            # SINGLE
            # =================================================

            single = run_single(
                question
            )

            single_correct = answers_match(
                single["final_answer"],
                truth,
            )

            # =================================================
            # AGENT FOREST
            # =================================================

            forest = run_agent_forest_3(
                question
            )

            forest_correct = answers_match(
                forest["final_answer"],
                truth,
            )

            # =================================================
            # SELF REFLECTION
            # =================================================

            reflection = run_self_reflection(
                question
            )

            reflection_correct = answers_match(
                reflection["final_answer"],
                truth,
            )

            reflection_draft_correct = answers_match(
                reflection["draft_answer"],
                truth,
            )

            reflection_positive = (
                not reflection_draft_correct
                and reflection_correct
            )

            reflection_negative = (
                reflection_draft_correct
                and not reflection_correct
            )

            # =================================================
            # PEER REVIEW
            # =================================================

            peer = run_peer_review(
                question
            )

            peer_correct = answers_match(
                peer["final_answer"],
                truth,
            )

            peer_writer_correct = answers_match(
                peer["writer_answer"],
                truth,
            )

            peer_reviewer_correct = answers_match(
                peer["reviewer_answer"],
                truth,
            )

            peer_positive = (
                not peer_writer_correct
                and peer_correct
            )

            peer_negative = (
                peer_writer_correct
                and not peer_correct
            )

            # =================================================
            # PROCESS EVENT CLASSIFICATION
            # =================================================

            if (
                not peer_writer_correct
                and peer_reviewer_correct
                and peer_correct
            ):
                peer_event = (
                    "successful_peer_correction"
                )

            elif (
                not peer_writer_correct
                and not peer_reviewer_correct
                and peer_correct
            ):
                peer_event = (
                    "editor_recovery"
                )

            elif (
                peer_writer_correct
                and not peer_reviewer_correct
                and peer_correct
            ):
                peer_event = (
                    "editor_rejects_bad_review"
                )

            elif (
                peer_writer_correct
                and not peer_reviewer_correct
                and not peer_correct
            ):
                peer_event = (
                    "harmful_review"
                )

            elif (
                not peer_writer_correct
                and peer_reviewer_correct
                and not peer_correct
            ):
                peer_event = (
                    "editor_rejects_useful_review"
                )

            elif (
                peer_writer_correct
                and peer_reviewer_correct
                and peer_correct
            ):
                peer_event = (
                    "stable_success"
                )

            else:
                peer_event = (
                    "other_or_persistent_failure"
                )

            # =================================================
            # ROW
            # =================================================

            row = {
                "repetition":
                    repetition,

                "question_id":
                    item["question_id"],

                "dataset_index":
                    item["dataset_index"],

                "level":
                    item["level"],

                "type":
                    item["type"],

                "ground_truth":
                    truth,

                # Single
                "single_answer":
                    single["final_answer"],

                "single_correct":
                    single_correct,

                "single_tokens":
                    single["total_tokens"],

                "single_wall_time":
                    single["wall_time"],

                # Forest
                "forest_answer":
                    forest["final_answer"],

                "forest_correct":
                    forest_correct,

                "forest_disagreement":
                    forest["disagreement"],

                "forest_tokens":
                    forest["total_tokens"],

                "forest_wall_time":
                    forest["wall_time"],

                # Reflection
                "reflection_draft_answer":
                    reflection["draft_answer"],

                "reflection_final_answer":
                    reflection["final_answer"],

                "reflection_draft_correct":
                    reflection_draft_correct,

                "reflection_correct":
                    reflection_correct,

                "reflection_positive_correction":
                    reflection_positive,

                "reflection_negative_correction":
                    reflection_negative,

                "reflection_critique_changed":
                    reflection[
                        "critique_changed_answer"
                    ],

                "reflection_tokens":
                    reflection["total_tokens"],

                "reflection_wall_time":
                    reflection["wall_time"],

                # Peer Review
                "peer_writer_answer":
                    peer["writer_answer"],

                "peer_reviewer_answer":
                    peer["reviewer_answer"],

                "peer_final_answer":
                    peer["final_answer"],

                "peer_writer_correct":
                    peer_writer_correct,

                "peer_reviewer_correct":
                    peer_reviewer_correct,

                "peer_correct":
                    peer_correct,

                "peer_positive_correction":
                    peer_positive,

                "peer_negative_correction":
                    peer_negative,

                "peer_reviewer_changed_writer":
                    peer["reviewer_changed_writer"],

                "peer_editor_changed_writer":
                    peer["editor_changed_writer"],

                "peer_editor_agreed_reviewer":
                    peer[
                        "editor_agreed_with_reviewer"
                    ],

                "peer_event":
                    peer_event,

                "peer_tokens":
                    peer["total_tokens"],

                "peer_wall_time":
                    peer["wall_time"],
            }

            all_rows.append(row)
            repetition_rows.append(row)

            print(
                f"  GT           : {truth}"
            )

            print(
                f"  Single       : "
                f"{single['final_answer']} "
                f"{'OK' if single_correct else 'ERR'}"
            )

            print(
                f"  Forest N=3   : "
                f"{forest['final_answer']} "
                f"{'OK' if forest_correct else 'ERR'}"
            )

            print(
                f"  Reflection   : "
                f"{reflection['final_answer']} "
                f"{'OK' if reflection_correct else 'ERR'}"
            )

            print(
                f"  Peer Review  : "
                f"{peer['final_answer']} "
                f"{'OK' if peer_correct else 'ERR'}"
            )

            print(
                f"  Peer event   : "
                f"{peer_event}"
            )

            # Save continuously
            save_json(
                os.path.join(
                    RESULTS_DIR,
                    "raw_results.json",
                ),
                all_rows,
            )

            save_csv(
                os.path.join(
                    RESULTS_DIR,
                    "raw_results.csv",
                ),
                all_rows,
            )

            torch.cuda.empty_cache()

        # =====================================================
        # REPETITION SUMMARY
        # =====================================================

        total = len(
            repetition_rows
        )

        def accuracy(field):
            return (
                sum(
                    row[field]
                    for row
                    in repetition_rows
                )
                / total
            )

        single_accuracy = accuracy(
            "single_correct"
        )

        forest_accuracy = accuracy(
            "forest_correct"
        )

        reflection_accuracy = accuracy(
            "reflection_correct"
        )

        peer_accuracy = accuracy(
            "peer_correct"
        )

        reflection_positive = sum(
            row[
                "reflection_positive_correction"
            ]
            for row in repetition_rows
        )

        reflection_negative = sum(
            row[
                "reflection_negative_correction"
            ]
            for row in repetition_rows
        )

        peer_positive = sum(
            row["peer_positive_correction"]
            for row in repetition_rows
        )

        peer_negative = sum(
            row["peer_negative_correction"]
            for row in repetition_rows
        )

        summary = {
            "repetition":
                repetition,

            "model":
                MODEL_NAME,

            "questions":
                total,

            "single_accuracy":
                single_accuracy,

            "forest_accuracy":
                forest_accuracy,

            "self_reflection_accuracy":
                reflection_accuracy,

            "peer_review_accuracy":
                peer_accuracy,

            "reflection_positive_corrections":
                reflection_positive,

            "reflection_negative_corrections":
                reflection_negative,

            "peer_positive_corrections":
                peer_positive,

            "peer_negative_corrections":
                peer_negative,

            "single_total_tokens":
                sum(
                    row["single_tokens"] or 0
                    for row
                    in repetition_rows
                ),

            "forest_total_tokens":
                sum(
                    row["forest_tokens"] or 0
                    for row
                    in repetition_rows
                ),

            "reflection_total_tokens":
                sum(
                    row["reflection_tokens"] or 0
                    for row
                    in repetition_rows
                ),

            "peer_total_tokens":
                sum(
                    row["peer_tokens"] or 0
                    for row
                    in repetition_rows
                ),
        }

        summaries.append(summary)

        save_json(
            os.path.join(
                RESULTS_DIR,
                "summaries.json",
            ),
            summaries,
        )

        save_csv(
            os.path.join(
                RESULTS_DIR,
                "summaries.csv",
            ),
            summaries,
        )

        print()
        print("=" * 76)
        print(
            f"SUMMARY — REPETITION {repetition}"
        )
        print("=" * 76)

        print(
            f"Single Writer      : "
            f"{single_accuracy:.2%}"
        )

        print(
            f"Agent Forest N=3  : "
            f"{forest_accuracy:.2%}"
        )

        print(
            f"Self-Reflection   : "
            f"{reflection_accuracy:.2%}"
        )

        print(
            f"Peer Review       : "
            f"{peer_accuracy:.2%}"
        )

        print()
        print(
            "Reflection corrections:",
            f"+{reflection_positive}",
            f"-{reflection_negative}",
        )

        print(
            "Peer Review corrections:",
            f"+{peer_positive}",
            f"-{peer_negative}",
        )

    # =========================================================
    # FINAL MEAN SUMMARY
    # =========================================================

    print()
    print("=" * 76)
    print("FINAL MEAN RESULTS")
    print("=" * 76)

    fields = [
        "single_accuracy",
        "forest_accuracy",
        "self_reflection_accuracy",
        "peer_review_accuracy",
    ]

    labels = [
        "Single Writer",
        "Agent Forest N=3",
        "Self-Reflection",
        "Peer Review",
    ]

    final_summary = {}

    for field, label in zip(
        fields,
        labels,
    ):
        values = [
            summary[field]
            for summary
            in summaries
        ]

        mean_value = (
            sum(values)
            / len(values)
        )

        final_summary[field] = {
            "values": values,
            "mean": mean_value,
        }

        print(
            f"{label:<18}: "
            f"{mean_value:.2%}"
        )

    save_json(
        os.path.join(
            RESULTS_DIR,
            "final_summary.json",
        ),
        final_summary,
    )

    print()
    print(
        "Results saved to:",
        RESULTS_DIR,
    )


if __name__ == "__main__":
    main()