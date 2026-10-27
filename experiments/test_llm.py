import re
import time
from collections import Counter

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_PATH = (
    "/home/isa1909/projects/def-sigmagao/"
    "isa1909/models/Qwen2.5-7B-Instruct"
)

# Initial small benchmark.
# Later we will replace this with GSM8K.
QUESTIONS = [
    ("What is 17 * 24?", "408"),
    ("What is 37 * 19?", "703"),
    ("What is 144 / 12?", "12"),
    ("What is 25 * 16?", "400"),
    ("What is 81 / 9?", "9"),
    ("What is 14 * 18?", "252"),
    ("What is 125 + 376?", "501"),
    ("What is 900 - 437?", "463"),
    ("What is 32 * 11?", "352"),
    ("What is 225 / 15?", "15"),

    (
        "A store has 24 boxes with 18 pencils in each box. "
        "How many pencils are there in total?",
        "432",
    ),
    (
        "Maria has 120 dollars. She spends 35 dollars and "
        "then receives 20 dollars. How much money does she have?",
        "105",
    ),
    (
        "A train travels 60 km per hour for 3 hours. "
        "How far does it travel?",
        "180",
    ),
    (
        "A class has 48 students divided equally into 6 groups. "
        "How many students are in each group?",
        "8",
    ),
    (
        "John buys 7 notebooks costing 13 dollars each. "
        "What is the total cost?",
        "91",
    ),
]


def extract_answer(text):
    """
    Extract the numeric value after FINAL:.

    Falls back to the last number in the response if the model
    does not follow the requested format.
    """

    match = re.search(
        r"FINAL:\s*(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )

    if match:
        return normalize_number(match.group(1))

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not numbers:
        return None

    return normalize_number(numbers[-1])


def normalize_number(value):
    """
    Normalize values such as:
        408
        408.0

    into the same representation.
    """

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(number)

    except (TypeError, ValueError):
        return str(value)


def build_prompts(tokenizer):
    prompts = []

    for question, _ in QUESTIONS:

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
                    "Solve the problem carefully. "
                    "You may reason step by step, but the final line "
                    "must contain only the answer in this format:\n"
                    "FINAL: <number>\n\n"
                    f"Problem: {question}"
                ),
            },
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts.append(text)

    return prompts


def majority_vote(answers):
    valid = [
        answer
        for answer in answers
        if answer is not None
    ]

    if not valid:
        return None

    counts = Counter(valid)

    return counts.most_common(1)[0][0]


def run_agent_forest(
    llm,
    prompts,
    n_agents,
):
    """
    Same model + same problem,
    independently sampled N times.

    Final answer = majority vote.
    """

    params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
        n=n_agents,
        seed=42,
    )

    start = time.time()

    outputs = llm.generate(
        prompts,
        params,
    )

    elapsed = time.time() - start

    results = []

    total_tokens = 0

    for (question, truth), output in zip(
        QUESTIONS,
        outputs,
    ):

        raw_responses = []
        answers = []

        for candidate in output.outputs:

            raw_text = candidate.text.strip()

            answer = extract_answer(raw_text)

            raw_responses.append(raw_text)
            answers.append(answer)

            total_tokens += len(
                candidate.token_ids
            )

        voted_answer = majority_vote(
            answers
        )

        correct = voted_answer == truth

        results.append(
            {
                "question": question,
                "truth": truth,
                "answers": answers,
                "voted_answer": voted_answer,
                "correct": correct,
                "raw_responses": raw_responses,
            }
        )

    return results, elapsed, total_tokens


def print_results(
    n_agents,
    results,
    elapsed,
    total_tokens,
):
    correct = sum(
        result["correct"]
        for result in results
    )

    accuracy = (
        correct / len(results)
    )

    total_generations = (
        len(results) * n_agents
    )

    print(
        "\n"
        + "=" * 70,
        flush=True,
    )

    print(
        f"AGENT FOREST N = {n_agents}",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    for i, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nQuestion {i}: "
            f"{result['question']}",
            flush=True,
        )

        print(
            f"Ground truth: "
            f"{result['truth']}",
            flush=True,
        )

        print(
            f"Individual answers: "
            f"{result['answers']}",
            flush=True,
        )

        print(
            f"Majority vote: "
            f"{result['voted_answer']}",
            flush=True,
        )

        print(
            f"Correct: "
            f"{result['correct']}",
            flush=True,
        )

    print(
        "\n--- SUMMARY ---",
        flush=True,
    )

    print(
        f"N agents: {n_agents}",
        flush=True,
    )

    print(
        f"Questions: "
        f"{len(results)}",
        flush=True,
    )

    print(
        f"Total generations: "
        f"{total_generations}",
        flush=True,
    )

    print(
        f"Correct: {correct}",
        flush=True,
    )

    print(
        f"Accuracy: "
        f"{accuracy:.2%}",
        flush=True,
    )

    print(
        f"Inference time: "
        f"{elapsed:.2f} seconds",
        flush=True,
    )

    print(
        f"Generated tokens: "
        f"{total_tokens}",
        flush=True,
    )

    print(
        f"Token throughput: "
        f"{total_tokens / elapsed:.2f} "
        f"tokens/sec",
        flush=True,
    )

    print(
        f"Generation throughput: "
        f"{total_generations / elapsed:.2f} "
        f"generations/sec",
        flush=True,
    )

    return accuracy


def compare_with_single(
    baseline_results,
    forest_results,
):
    positive_corrections = 0
    negative_corrections = 0

    for baseline, forest in zip(
        baseline_results,
        forest_results,
    ):

        # Single agent wrong -> forest correct
        if (
            not baseline["correct"]
            and forest["correct"]
        ):
            positive_corrections += 1

        # Single agent correct -> forest wrong
        if (
            baseline["correct"]
            and not forest["correct"]
        ):
            negative_corrections += 1

    return (
        positive_corrections,
        negative_corrections,
    )


def main():

    print(
        "=== Initial Agent Forest Experiment ===",
        flush=True,
    )

    print(
        f"Model: {MODEL_PATH}",
        flush=True,
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    prompts = build_prompts(
        tokenizer
    )

    # --------------------------------------------------
    # Load Qwen only ONCE
    # --------------------------------------------------

    print(
        "\nLoading model with vLLM...",
        flush=True,
    )

    load_start = time.time()

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )

    load_time = (
        time.time() - load_start
    )

    print(
        f"Model loaded in "
        f"{load_time:.2f} seconds",
        flush=True,
    )

    # --------------------------------------------------
    # Agent Forest sizes
    # --------------------------------------------------

    forest_sizes = [
        1,
        3,
        5,
        10,
    ]

    all_results = {}
    accuracies = {}

    for n_agents in forest_sizes:

        results, elapsed, tokens = (
            run_agent_forest(
                llm,
                prompts,
                n_agents,
            )
        )

        accuracy = print_results(
            n_agents,
            results,
            elapsed,
            tokens,
        )

        all_results[n_agents] = results
        accuracies[n_agents] = accuracy

    # --------------------------------------------------
    # Comparison with N=1
    # --------------------------------------------------

    baseline = all_results[1]

    print(
        "\n"
        + "=" * 70,
        flush=True,
    )

    print(
        "FINAL COMPARISON",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    for n_agents in forest_sizes:

        positive, negative = (
            compare_with_single(
                baseline,
                all_results[n_agents],
            )
        )

        print(
            f"N={n_agents:2d} | "
            f"accuracy="
            f"{accuracies[n_agents]:.2%} | "
            f"positive corrections="
            f"{positive} | "
            f"negative corrections="
            f"{negative}",
            flush=True,
        )


if __name__ == "__main__":
    main()