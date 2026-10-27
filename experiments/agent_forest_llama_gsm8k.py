import json
import re
import time
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = (
    "/home/isa1909/projects/def-sigmagao/"
    "isa1909/models/Llama-2-13b-chat-hf"
)

N_QUESTIONS = 20

FOREST_SIZES = [
    1,
    3,
    5,
]

DATASET_SEED = 42
TEMPERATURE = 1.0
TOP_P = 1.0

MAX_NEW_TOKENS = 256

RESULTS_DIR = Path("results/llama2_gsm8k_test")


# ============================================================
# ANSWER PARSING
# ============================================================

def normalize_number(value):
    if value is None:
        return None

    value = str(value).strip()
    value = value.replace(",", "")

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(number)

    except ValueError:
        return value


def extract_model_answer(text):
    """
    Preferred format:
        FINAL: 123

    Fallback:
        use the last numeric value in the response.
    """

    match = re.search(
        r"FINAL:\s*(-?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )

    if match:
        return normalize_number(match.group(1))

    numbers = re.findall(
        r"-?[\d,]+(?:\.\d+)?",
        text,
    )

    if not numbers:
        return None

    return normalize_number(numbers[-1])


def extract_gsm8k_ground_truth(answer_text):
    """
    GSM8K answers normally end with:
        #### 42
    """

    match = re.search(
        r"####\s*(-?[\d,]+(?:\.\d+)?)",
        answer_text,
    )

    if not match:
        raise ValueError(
            f"Could not extract GSM8K answer from: {answer_text}"
        )

    return normalize_number(match.group(1))


# ============================================================
# DATASET
# ============================================================

def load_gsm8k_sample():

    print("Loading GSM8K...", flush=True)

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
    )

    dataset = dataset.shuffle(
        seed=DATASET_SEED
    )

    dataset = dataset.select(
        range(N_QUESTIONS)
    )

    questions = []

    for idx, row in enumerate(dataset):

        truth = extract_gsm8k_ground_truth(
            row["answer"]
        )

        questions.append(
            {
                "id": idx,
                "question": row["question"],
                "truth": truth,
            }
        )

    print(
        f"Loaded {len(questions)} questions.",
        flush=True,
    )

    return questions


# ============================================================
# PROMPTS
# ============================================================

def build_prompts(
    tokenizer,
    questions,
):

    prompts = []

    for item in questions:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful mathematical reasoning assistant."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Solve the following math problem carefully. "
                    "Reason step by step. "
                    "At the end, output exactly one line in the format:\n"
                    "FINAL: <number>\n\n"
                    f"Problem:\n{item['question']}"
                ),
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts.append(prompt)

    return prompts


# ============================================================
# AGENT FOREST
# ============================================================

def majority_vote(answers):

    valid_answers = [
        answer
        for answer in answers
        if answer is not None
    ]

    if not valid_answers:
        return None

    counts = Counter(valid_answers)

    return counts.most_common(1)[0][0]


def run_forest(
    llm,
    prompts,
    questions,
    n_agents,
):

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        n=n_agents,
    )

    print(
        f"\nRunning Agent Forest with N={n_agents}...",
        flush=True,
    )

    start = time.time()

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    elapsed = time.time() - start

    results = []

    total_generated_tokens = 0

    for question, output in zip(
        questions,
        outputs,
    ):

        answers = []
        raw_responses = []

        for candidate in output.outputs:

            raw_text = candidate.text.strip()

            parsed_answer = extract_model_answer(
                raw_text
            )

            answers.append(
                parsed_answer
            )

            raw_responses.append(
                raw_text
            )

            total_generated_tokens += len(
                candidate.token_ids
            )

        voted_answer = majority_vote(
            answers
        )

        correct = (
            voted_answer
            == question["truth"]
        )

        results.append(
            {
                "id": question["id"],
                "question": question["question"],
                "truth": question["truth"],
                "answers": answers,
                "voted_answer": voted_answer,
                "correct": correct,
                "raw_responses": raw_responses,
            }
        )

    return (
        results,
        elapsed,
        total_generated_tokens,
    )


# ============================================================
# REPORTING
# ============================================================

def summarize(
    n_agents,
    results,
    elapsed,
    total_tokens,
):

    correct = sum(
        item["correct"]
        for item in results
    )

    accuracy = (
        correct / len(results)
    )

    total_generations = (
        len(results) * n_agents
    )

    disagreement = sum(
        len(set(
            answer
            for answer in item["answers"]
            if answer is not None
        )) > 1
        for item in results
    )

    summary = {
        "n_agents": n_agents,
        "questions": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "disagreement_questions": disagreement,
        "elapsed_seconds": elapsed,
        "generated_tokens": total_tokens,
        "total_generations": total_generations,
    }

    print(
        "\n"
        + "=" * 60,
        flush=True,
    )

    print(
        f"N = {n_agents}",
        flush=True,
    )

    print(
        f"Accuracy: {accuracy:.2%}",
        flush=True,
    )

    print(
        f"Correct: {correct}/{len(results)}",
        flush=True,
    )

    print(
        f"Disagreement questions: {disagreement}",
        flush=True,
    )

    print(
        f"Generated tokens: {total_tokens}",
        flush=True,
    )

    print(
        f"Inference time: {elapsed:.2f}s",
        flush=True,
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== Llama2-13B Agent Forest / GSM8K ===",
        flush=True,
    )

    print(
        f"Model: {MODEL_PATH}",
        flush=True,
    )

    questions = load_gsm8k_sample()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    prompts = build_prompts(
        tokenizer,
        questions,
    )

    print(
        "\nLoading Llama2-13B with vLLM...",
        flush=True,
    )

    load_start = time.time()

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )

    print(
        f"Model loaded in "
        f"{time.time() - load_start:.2f}s",
        flush=True,
    )

    all_results = {}
    summaries = []

    for n_agents in FOREST_SIZES:

        results, elapsed, tokens = run_forest(
            llm,
            prompts,
            questions,
            n_agents,
        )

        summary = summarize(
            n_agents,
            results,
            elapsed,
            tokens,
        )

        all_results[str(n_agents)] = results
        summaries.append(summary)

    # --------------------------------------------------------
    # Positive / negative corrections relative to N=1
    # --------------------------------------------------------

    baseline = all_results["1"]

    for summary in summaries:

        n = summary["n_agents"]

        if n == 1:
            summary["positive_corrections"] = 0
            summary["negative_corrections"] = 0
            continue

        forest = all_results[str(n)]

        positive = 0
        negative = 0

        for single, multi in zip(
            baseline,
            forest,
        ):

            if (
                not single["correct"]
                and multi["correct"]
            ):
                positive += 1

            if (
                single["correct"]
                and not multi["correct"]
            ):
                negative += 1

        summary["positive_corrections"] = positive
        summary["negative_corrections"] = negative

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        RESULTS_DIR / "summaries.json",
        "w",
    ) as f:

        json.dump(
            summaries,
            f,
            indent=2,
        )

    with open(
        RESULTS_DIR / "raw_results.json",
        "w",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
        )

    print(
        "\n=== FINAL RESULTS ===",
        flush=True,
    )

    for summary in summaries:

        print(
            f"N={summary['n_agents']:2d} | "
            f"accuracy={summary['accuracy']:.2%} | "
            f"+corr={summary['positive_corrections']} | "
            f"-corr={summary['negative_corrections']} | "
            f"time={summary['elapsed_seconds']:.2f}s",
            flush=True,
        )

    print(
        f"\nResults saved to: {RESULTS_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()