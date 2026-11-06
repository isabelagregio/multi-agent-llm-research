#!/usr/bin/env python3

import os
import re
import json
import time
import random
import multiprocessing

from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from math_verify import (
        parse as math_parse,
        verify as math_verify,
        LatexExtractionConfig,
        ExprExtractionConfig,
    )
except ImportError as exc:
    raise SystemExit(
        "math-verify is required. Install once with:\n"
        "pip install 'math-verify[antlr4_13_2]'"
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        "/home/isa1909/projects/def-sigmagao/isa1909",
    )
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(
        PROJECT_ROOT
        / "models"
        / "Qwen2.5-7B-Instruct"
    ),
)

RESULTS_SUBDIR = os.environ.get(
    "RESULTS_SUBDIR",
    "experiment_5_single_forest_selfrefine_studentcritic",
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / RESULTS_SUBDIR
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

N_QUESTIONS = int(
    os.environ.get(
        "N_QUESTIONS",
        "200",
    )
)

N_REPETITIONS = int(
    os.environ.get(
        "N_REPETITIONS",
        "3",
    )
)

FOREST_N = int(
    os.environ.get(
        "FOREST_N",
        "5",
    )
)

REFINE_ROUNDS = int(
    os.environ.get(
        "REFINE_ROUNDS",
        "2",
    )
)

CRITIC_ROUNDS = int(
    os.environ.get(
        "CRITIC_ROUNDS",
        "2",
    )
)

TEMPERATURE = float(
    os.environ.get(
        "TEMPERATURE",
        "0.7",
    )
)

TOP_P = float(
    os.environ.get(
        "TOP_P",
        "0.95",
    )
)

MAX_NEW_TOKENS = int(
    os.environ.get(
        "MAX_NEW_TOKENS",
        "512",
    )
)

MAX_MODEL_LEN = int(
    os.environ.get(
        "MAX_MODEL_LEN",
        "8192",
    )
)

MAX_NUM_SEQS = int(
    os.environ.get(
        "MAX_NUM_SEQS",
        "64",
    )
)

GPU_MEMORY_UTILIZATION = float(
    os.environ.get(
        "GPU_MEMORY_UTILIZATION",
        "0.90",
    )
)

DATASET_SEED = int(
    os.environ.get(
        "DATASET_SEED",
        "42",
    )
)

GENERATION_SEED = int(
    os.environ.get(
        "GENERATION_SEED",
        "42",
    )
)


# ============================================================
# MATCHED BUDGET CHECK
# ============================================================

if 1 + 2 * REFINE_ROUNDS != FOREST_N:
    raise ValueError(
        "Require 1 + 2*REFINE_ROUNDS == FOREST_N "
        "for matched inference budget."
    )

if 1 + 2 * CRITIC_ROUNDS != FOREST_N:
    raise ValueError(
        "Require 1 + 2*CRITIC_ROUNDS == FOREST_N "
        "for matched inference budget."
    )


# ============================================================
# GLOBAL ENGINE VARIABLES
# ============================================================

tokenizer = None
llm = None


# ============================================================
# REGEX
# ============================================================

FINAL_RE = re.compile(
    r"FINAL:\s*(.+?)(?:\n|$)",
    flags=re.IGNORECASE,
)

VERDICT_RE = re.compile(
    r"VERDICT:\s*(ACCEPT|REJECT)",
    flags=re.IGNORECASE,
)


# ============================================================
# FILE HELPERS
# ============================================================

def save_json(path, data):
    tmp = str(path) + ".tmp"

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        tmp,
        path,
    )


def append_jsonl(path, rows):
    with open(
        path,
        "a",
        encoding="utf-8",
    ) as file:

        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# DATASET
# ============================================================

def load_math_sample():

    print(
        "Loading MATH dataset...",
        flush=True,
    )

    dataset = load_dataset(
        "jeggers/competition_math",
        "original",
        split="test",
    )

    if N_QUESTIONS > len(dataset):
        raise ValueError(
            f"N_QUESTIONS={N_QUESTIONS} "
            f"exceeds dataset size={len(dataset)}"
        )

    rng = random.Random(
        DATASET_SEED
    )

    indices = rng.sample(
        range(len(dataset)),
        N_QUESTIONS,
    )

    questions = []

    for question_id, dataset_index in enumerate(
        indices,
        start=1,
    ):

        row = dataset[
            dataset_index
        ]

        ground_truth = row.get(
            "extracted_solution"
        )

        if ground_truth is None:
            raise ValueError(
                "Dataset row has no extracted_solution."
            )

        questions.append(
            {
                "question_id":
                    question_id,

                "dataset_index":
                    dataset_index,

                "problem":
                    row["problem"],

                "gold":
                    str(ground_truth),

                "level":
                    row.get("level"),

                "type":
                    row.get("type"),
            }
        )

    print(
        f"Loaded {len(questions)} questions.",
        flush=True,
    )

    return questions


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_final_text(text):

    if not text:
        return None

    matches = FINAL_RE.findall(
        text
    )

    if matches:
        return matches[-1].strip()

    return text.strip()


# ============================================================
# MATH EQUIVALENCE
# ============================================================

def parse_math(text):

    if text is None:
        return []

    candidate = str(
        text
    ).strip()

    if not (
        (
            candidate.startswith("$")
            and candidate.endswith("$")
        )
        or "\\boxed" in candidate
    ):
        candidate = (
            f"${candidate}$"
        )

    return math_parse(
        candidate,
        extraction_config=[
            LatexExtractionConfig(),
            ExprExtractionConfig(),
        ],
    )


def answers_match(
    prediction,
    ground_truth,
):

    if prediction is None:
        return False

    try:
        parsed_gold = parse_math(
            ground_truth
        )

        parsed_prediction = parse_math(
            prediction
        )

        if (
            not parsed_gold
            or not parsed_prediction
        ):
            return False

        return bool(
            math_verify(
                parsed_gold,
                parsed_prediction,
            )
        )

    except Exception:
        return False


# ============================================================
# PROMPTS
# ============================================================

SOLVER_SYSTEM = """
You are a careful competition mathematics solver.

Solve the problem independently and show enough reasoning
to make the solution checkable.

Do not assume that another agent will correct you.

At the end, put only the final mathematical answer on a
separate line exactly in this format:

FINAL: <answer>
""".strip()


SELF_FEEDBACK_SYSTEM = """
You are reviewing your own previous solution to a competition
mathematics problem.

Critically inspect the reasoning step by step.

Look for:
- arithmetic errors;
- algebraic errors;
- logical errors;
- interpretation errors;
- missing cases.

Do not change a correct solution merely to be different.

Give concise and actionable feedback.
""".strip()


SELF_REFINE_SYSTEM = """
You are refining your own previous solution using self-feedback.

Use the original problem, previous solution, and feedback.

Check the feedback independently.

Produce the best corrected solution.

At the end, put only the final mathematical answer on a
separate line exactly in this format:

FINAL: <answer>
""".strip()


CRITIC_SYSTEM = """
You are an independent mathematical Critic evaluating a
Student's solution.

Verify the reasoning step by step.

Be conservative:
do not reject a correct solution only because you would
solve the problem differently.

If there is an error, identify the earliest important
incorrect step and explain why it is wrong.

Return exactly:

VERDICT: ACCEPT or REJECT
ERROR_STEP: <short description, or NONE>
CRITIQUE: <concise explanation>
SUGGESTION: <actionable correction, or NONE>
""".strip()


STUDENT_UPDATE_SYSTEM = """
You are the Student revising a mathematical solution after
receiving feedback from an independent Critic.

Check the Critic's feedback independently because the Critic
may also be wrong.

Keep a correct solution if the criticism is invalid.

Repair the solution if the Critic found a real error.

At the end, put only the final mathematical answer on a
separate line exactly in this format:

FINAL: <answer>
""".strip()


# ============================================================
# MESSAGE BUILDERS
# ============================================================

def build_messages(
    system_prompt,
    user_prompt,
):

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def solver_messages(problem):

    return build_messages(
        SOLVER_SYSTEM,
        problem,
    )


def self_feedback_messages(
    problem,
    previous_solution,
):

    prompt = f"""
ORIGINAL PROBLEM

{problem}

==================================================

PREVIOUS SOLUTION

{previous_solution}
""".strip()

    return build_messages(
        SELF_FEEDBACK_SYSTEM,
        prompt,
    )


def self_refine_messages(
    problem,
    previous_solution,
    feedback,
):

    prompt = f"""
ORIGINAL PROBLEM

{problem}

==================================================

PREVIOUS SOLUTION

{previous_solution}

==================================================

SELF-FEEDBACK

{feedback}
""".strip()

    return build_messages(
        SELF_REFINE_SYSTEM,
        prompt,
    )


def critic_messages(
    problem,
    previous_solution,
):

    prompt = f"""
ORIGINAL PROBLEM

{problem}

==================================================

STUDENT SOLUTION

{previous_solution}
""".strip()

    return build_messages(
        CRITIC_SYSTEM,
        prompt,
    )


def student_update_messages(
    problem,
    previous_solution,
    critique,
):

    prompt = f"""
ORIGINAL PROBLEM

{problem}

==================================================

CURRENT STUDENT SOLUTION

{previous_solution}

==================================================

CRITIC FEEDBACK

{critique}
""".strip()

    return build_messages(
        STUDENT_UPDATE_SYSTEM,
        prompt,
    )


# ============================================================
# CRITIC VERDICT
# ============================================================

def parse_verdict(text):

    if not text:
        return "UNKNOWN"

    match = VERDICT_RE.search(
        text
    )

    if not match:
        return "UNKNOWN"

    return match.group(1).upper()


# ============================================================
# ENGINE INITIALIZATION
# ============================================================

def initialize_engine():

    global tokenizer
    global llm

    print(
        "Loading tokenizer...",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    print(
        "Loading vLLM engine once...",
        flush=True,
    )

    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=(
            GPU_MEMORY_UTILIZATION
        ),
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    print(
        "vLLM engine ready.",
        flush=True,
    )


# ============================================================
# PROMPT RENDERING
# ============================================================

def render_messages(messages):

    if tokenizer is None:
        raise RuntimeError(
            "Tokenizer not initialized."
        )

    prompt = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    token_count = len(
        tokenizer(
            prompt,
            add_special_tokens=False,
        ).input_ids
    )

    if (
        token_count
        + MAX_NEW_TOKENS
        > MAX_MODEL_LEN
    ):
        raise ValueError(
            f"Prompt too long: "
            f"{token_count} input + "
            f"{MAX_NEW_TOKENS} output "
            f"> {MAX_MODEL_LEN}"
        )

    return prompt


# ============================================================
# SAMPLING
# ============================================================

def make_sampling_params(
    seed,
    n=1,
):

    return SamplingParams(
        n=n,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        seed=seed,
    )


# ============================================================
# BATCHED GENERATION
# ============================================================

def generate_batch(
    messages_batch,
    seed,
    n=1,
):

    if llm is None:
        raise RuntimeError(
            "vLLM engine not initialized."
        )

    prompts = [
        render_messages(messages)
        for messages
        in messages_batch
    ]

    params = make_sampling_params(
        seed=seed,
        n=n,
    )

    start = time.perf_counter()

    outputs = llm.generate(
        prompts,
        params,
        use_tqdm=True,
    )

    wall_time = (
        time.perf_counter()
        - start
    )

    parsed = []

    for request_output in outputs:

        prompt_tokens = len(
            getattr(
                request_output,
                "prompt_token_ids",
                [],
            )
            or []
        )

        candidates = []

        for candidate in (
            request_output.outputs
        ):

            completion_tokens = len(
                candidate.token_ids
            )

            candidates.append(
                {
                    "text":
                        candidate.text,

                    "answer":
                        extract_final_text(
                            candidate.text
                        ),

                    "prompt_tokens":
                        prompt_tokens,

                    "completion_tokens":
                        completion_tokens,

                    "total_tokens":
                        (
                            prompt_tokens
                            + completion_tokens
                        ),

                    "cumulative_logprob":
                        getattr(
                            candidate,
                            "cumulative_logprob",
                            None,
                        ),
                }
            )

        parsed.append(
            candidates
        )

    return (
        parsed,
        wall_time,
    )


# ============================================================
# MAJORITY / PLURALITY VOTE
# ============================================================

def vote_key(answer):

    if answer is None:
        return None

    return re.sub(
        r"\s+",
        "",
        str(answer)
        .strip()
        .lower(),
    )


def plurality_vote(candidates):

    valid = [
        candidate
        for candidate
        in candidates
        if candidate["answer"]
        is not None
    ]

    if not valid:
        return (
            None,
            True,
            {},
        )

    groups = defaultdict(
        list
    )

    for candidate in valid:

        key = vote_key(
            candidate["answer"]
        )

        groups[key].append(
            candidate
        )

    counts = {
        key: len(group)
        for key, group
        in groups.items()
    }

    max_count = max(
        counts.values()
    )

    tied_keys = [
        key
        for key, count
        in counts.items()
        if count == max_count
    ]

    tie = (
        len(tied_keys)
        > 1
    )

    if len(tied_keys) == 1:

        winner_key = (
            tied_keys[0]
        )

    else:

        def group_score(key):

            scores = [
                candidate[
                    "cumulative_logprob"
                ]
                for candidate
                in groups[key]
                if candidate[
                    "cumulative_logprob"
                ]
                is not None
            ]

            if not scores:
                return float("-inf")

            return max(
                scores
            )

        winner_key = max(
            tied_keys,
            key=group_score,
        )

    winner = (
        groups[
            winner_key
        ][0]["answer"]
    )

    return (
        winner,
        tie,
        counts,
    )


# ============================================================
# CORRECTION METRICS
# ============================================================

def correction_flags(
    initial_correct,
    final_correct,
):

    return {
        "positive_correction":
            (
                not initial_correct
                and final_correct
            ),

        "negative_correction":
            (
                initial_correct
                and not final_correct
            ),
    }


def accuracy(
    rows,
    field,
):

    if not rows:
        return None

    return (
        sum(
            bool(row[field])
            for row
            in rows
        )
        / len(rows)
    )


# ============================================================
# CRITIC QUALITY METRICS
# ============================================================

def critic_metrics(records):

    true_reject = 0
    false_reject = 0
    true_accept = 0
    false_accept = 0
    unknown = 0

    for record in records:

        verdict = record[
            "verdict"
        ]

        correct = record[
            "student_correct_before"
        ]

        if verdict == "UNKNOWN":

            unknown += 1

        elif (
            verdict == "REJECT"
            and not correct
        ):

            true_reject += 1

        elif (
            verdict == "REJECT"
            and correct
        ):

            false_reject += 1

        elif (
            verdict == "ACCEPT"
            and correct
        ):

            true_accept += 1

        elif (
            verdict == "ACCEPT"
            and not correct
        ):

            false_accept += 1

    precision = (
        true_reject
        /
        (
            true_reject
            + false_reject
        )
        if (
            true_reject
            + false_reject
        ) > 0
        else None
    )

    recall = (
        true_reject
        /
        (
            true_reject
            + false_accept
        )
        if (
            true_reject
            + false_accept
        ) > 0
        else None
    )

    f1 = (
        2
        * precision
        * recall
        /
        (
            precision
            + recall
        )
        if (
            precision is not None
            and recall is not None
            and (
                precision
                + recall
            ) > 0
        )
        else None
    )

    return {
        "true_reject":
            true_reject,

        "false_reject":
            false_reject,

        "true_accept":
            true_accept,

        "false_accept":
            false_accept,

        "unknown_verdict":
            unknown,

        "error_detection_precision":
            precision,

        "error_detection_recall":
            recall,

        "error_detection_f1":
            f1,
    }


# ============================================================
# ONE REPETITION
# ============================================================

def run_repetition(
    questions,
    repetition,
):

    repetition_seed = (
        GENERATION_SEED
        + 10000
        * repetition
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"REPETITION "
        f"{repetition}/"
        f"{N_REPETITIONS}"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # SHARED INITIAL RESPONSE
    # ========================================================

    print(
        "\n[1] Shared initial solutions",
        flush=True,
    )

    initial_outputs, single_time = (
        generate_batch(
            [
                solver_messages(
                    question["problem"]
                )
                for question
                in questions
            ],
            seed=(
                repetition_seed
                + 1
            ),
            n=1,
        )
    )

    initial = [
        output[0]
        for output
        in initial_outputs
    ]


    # ========================================================
    # AGENT FOREST
    # ========================================================

    print(
        f"\n[2] Agent Forest N={FOREST_N}",
        flush=True,
    )

    forest_extra_outputs, forest_time = (
        generate_batch(
            [
                solver_messages(
                    question["problem"]
                )
                for question
                in questions
            ],
            seed=(
                repetition_seed
                + 2
            ),
            n=(
                FOREST_N
                - 1
            ),
        )
    )

    forest_data = []

    for index in range(
        len(questions)
    ):

        candidates = (
            [initial[index]]
            +
            forest_extra_outputs[
                index
            ]
        )

        (
            final_answer,
            vote_tie,
            vote_counts,
        ) = plurality_vote(
            candidates
        )

        forest_data.append(
            {
                "candidates":
                    candidates,

                "final_answer":
                    final_answer,

                "vote_tie":
                    vote_tie,

                "vote_counts":
                    vote_counts,
            }
        )


    # ========================================================
    # SELF-REFINE
    # ========================================================

    self_current_text = [
        item["text"]
        for item
        in initial
    ]

    self_rounds = [
        []
        for _
        in questions
    ]

    self_time = 0.0

    for round_id in range(
        1,
        REFINE_ROUNDS + 1,
    ):

        print(
            f"\n[3] Self-Refine feedback "
            f"round {round_id}",
            flush=True,
        )

        feedback_outputs, duration = (
            generate_batch(
                [
                    self_feedback_messages(
                        questions[
                            index
                        ]["problem"],
                        self_current_text[
                            index
                        ],
                    )
                    for index
                    in range(
                        len(questions)
                    )
                ],
                seed=(
                    repetition_seed
                    + 100
                    + 2
                    * round_id
                ),
                n=1,
            )
        )

        self_time += duration

        feedback = [
            output[0]
            for output
            in feedback_outputs
        ]

        print(
            f"\n[4] Self-Refine revision "
            f"round {round_id}",
            flush=True,
        )

        refine_outputs, duration = (
            generate_batch(
                [
                    self_refine_messages(
                        questions[
                            index
                        ]["problem"],
                        self_current_text[
                            index
                        ],
                        feedback[
                            index
                        ]["text"],
                    )
                    for index
                    in range(
                        len(questions)
                    )
                ],
                seed=(
                    repetition_seed
                    + 101
                    + 2
                    * round_id
                ),
                n=1,
            )
        )

        self_time += duration

        refined = [
            output[0]
            for output
            in refine_outputs
        ]

        for index in range(
            len(questions)
        ):

            self_rounds[
                index
            ].append(
                {
                    "round":
                        round_id,

                    "feedback":
                        feedback[
                            index
                        ],

                    "refined":
                        refined[
                            index
                        ],
                }
            )

        self_current_text = [
            item["text"]
            for item
            in refined
        ]

    self_final = [
        rounds[-1][
            "refined"
        ]
        for rounds
        in self_rounds
    ]


    # ========================================================
    # STUDENT-CRITIC
    # ========================================================

    critic_current_text = [
        item["text"]
        for item
        in initial
    ]

    critic_rounds = [
        []
        for _
        in questions
    ]

    critic_records = []

    critic_time = 0.0

    for round_id in range(
        1,
        CRITIC_ROUNDS + 1,
    ):

        print(
            f"\n[5] Critic round "
            f"{round_id}",
            flush=True,
        )

        critic_outputs, duration = (
            generate_batch(
                [
                    critic_messages(
                        questions[
                            index
                        ]["problem"],
                        critic_current_text[
                            index
                        ],
                    )
                    for index
                    in range(
                        len(questions)
                    )
                ],
                seed=(
                    repetition_seed
                    + 200
                    + 2
                    * round_id
                ),
                n=1,
            )
        )

        critic_time += duration

        critiques = [
            output[0]
            for output
            in critic_outputs
        ]

        before_answers = [
            extract_final_text(
                text
            )
            for text
            in critic_current_text
        ]

        before_correct = [
            answers_match(
                before_answers[
                    index
                ],
                questions[
                    index
                ]["gold"],
            )
            for index
            in range(
                len(questions)
            )
        ]

        print(
            f"\n[6] Student update "
            f"round {round_id}",
            flush=True,
        )

        update_outputs, duration = (
            generate_batch(
                [
                    student_update_messages(
                        questions[
                            index
                        ]["problem"],
                        critic_current_text[
                            index
                        ],
                        critiques[
                            index
                        ]["text"],
                    )
                    for index
                    in range(
                        len(questions)
                    )
                ],
                seed=(
                    repetition_seed
                    + 201
                    + 2
                    * round_id
                ),
                n=1,
            )
        )

        critic_time += duration

        updated = [
            output[0]
            for output
            in update_outputs
        ]

        for index in range(
            len(questions)
        ):

            verdict = parse_verdict(
                critiques[
                    index
                ]["text"]
            )

            after_correct = (
                answers_match(
                    updated[
                        index
                    ]["answer"],
                    questions[
                        index
                    ]["gold"],
                )
            )

            round_record = {
                "round":
                    round_id,

                "verdict":
                    verdict,

                "critique":
                    critiques[
                        index
                    ],

                "student_before_answer":
                    before_answers[
                        index
                    ],

                "student_correct_before":
                    before_correct[
                        index
                    ],

                "student_after":
                    updated[
                        index
                    ],

                "student_correct_after":
                    after_correct,
            }

            critic_rounds[
                index
            ].append(
                round_record
            )

            critic_records.append(
                {
                    "round":
                        round_id,

                    "verdict":
                        verdict,

                    "student_correct_before":
                        before_correct[
                            index
                        ],

                    "student_correct_after":
                        after_correct,
                }
            )

        critic_current_text = [
            item["text"]
            for item
            in updated
        ]

    critic_final = [
        rounds[-1][
            "student_after"
        ]
        for rounds
        in critic_rounds
    ]


    # ========================================================
    # SCORE RESULTS
    # ========================================================

    rows = []

    for index, question in enumerate(
        questions
    ):

        single_correct = (
            answers_match(
                initial[
                    index
                ]["answer"],
                question["gold"],
            )
        )

        forest_correct = (
            answers_match(
                forest_data[
                    index
                ]["final_answer"],
                question["gold"],
            )
        )

        self_correct = (
            answers_match(
                self_final[
                    index
                ]["answer"],
                question["gold"],
            )
        )

        critic_correct = (
            answers_match(
                critic_final[
                    index
                ]["answer"],
                question["gold"],
            )
        )

        forest_correction = (
            correction_flags(
                single_correct,
                forest_correct,
            )
        )

        self_correction = (
            correction_flags(
                single_correct,
                self_correct,
            )
        )

        critic_correction = (
            correction_flags(
                single_correct,
                critic_correct,
            )
        )

        forest_tokens = sum(
            candidate[
                "total_tokens"
            ]
            for candidate
            in forest_data[
                index
            ]["candidates"]
        )

        self_tokens = (
            initial[
                index
            ]["total_tokens"]
            +
            sum(
                round_item[
                    "feedback"
                ]["total_tokens"]
                +
                round_item[
                    "refined"
                ]["total_tokens"]
                for round_item
                in self_rounds[
                    index
                ]
            )
        )

        critic_tokens = (
            initial[
                index
            ]["total_tokens"]
            +
            sum(
                round_item[
                    "critique"
                ]["total_tokens"]
                +
                round_item[
                    "student_after"
                ]["total_tokens"]
                for round_item
                in critic_rounds[
                    index
                ]
            )
        )

        rows.append(
            {
                "repetition":
                    repetition,

                "question_id":
                    question[
                        "question_id"
                    ],

                "dataset_index":
                    question[
                        "dataset_index"
                    ],

                "level":
                    question[
                        "level"
                    ],

                "type":
                    question[
                        "type"
                    ],

                "gold":
                    question[
                        "gold"
                    ],

                "single_answer":
                    initial[
                        index
                    ]["answer"],

                "single_correct":
                    single_correct,

                "single_tokens":
                    initial[
                        index
                    ]["total_tokens"],

                "forest_answer":
                    forest_data[
                        index
                    ]["final_answer"],

                "forest_correct":
                    forest_correct,

                "forest_positive_correction":
                    forest_correction[
                        "positive_correction"
                    ],

                "forest_negative_correction":
                    forest_correction[
                        "negative_correction"
                    ],

                "forest_vote_tie":
                    forest_data[
                        index
                    ]["vote_tie"],

                "forest_vote_counts":
                    forest_data[
                        index
                    ]["vote_counts"],

                "forest_tokens":
                    forest_tokens,

                "self_refine_answer":
                    self_final[
                        index
                    ]["answer"],

                "self_refine_correct":
                    self_correct,

                "self_refine_positive_correction":
                    self_correction[
                        "positive_correction"
                    ],

                "self_refine_negative_correction":
                    self_correction[
                        "negative_correction"
                    ],

                "self_refine_tokens":
                    self_tokens,

                "student_critic_answer":
                    critic_final[
                        index
                    ]["answer"],

                "student_critic_correct":
                    critic_correct,

                "student_critic_positive_correction":
                    critic_correction[
                        "positive_correction"
                    ],

                "student_critic_negative_correction":
                    critic_correction[
                        "negative_correction"
                    ],

                "student_critic_tokens":
                    critic_tokens,

                "single_trace":
                    initial[
                        index
                    ],

                "forest_trace":
                    forest_data[
                        index
                    ],

                "self_refine_trace":
                    self_rounds[
                        index
                    ],

                "student_critic_trace":
                    critic_rounds[
                        index
                    ],
            }
        )


    # ========================================================
    # SUMMARY HELPERS
    # ========================================================

    def correction_summary(prefix):

        positive = sum(
            row[
                f"{prefix}_positive_correction"
            ]
            for row
            in rows
        )

        negative = sum(
            row[
                f"{prefix}_negative_correction"
            ]
            for row
            in rows
        )

        baseline_wrong = sum(
            not row[
                "single_correct"
            ]
            for row
            in rows
        )

        baseline_right = sum(
            row[
                "single_correct"
            ]
            for row
            in rows
        )

        return {
            "positive_corrections":
                positive,

            "negative_corrections":
                negative,

            "positive_correction_rate":
                (
                    positive
                    /
                    baseline_wrong
                    if baseline_wrong
                    else None
                ),

            "negative_correction_rate":
                (
                    negative
                    /
                    baseline_right
                    if baseline_right
                    else None
                ),
        }


    # ========================================================
    # REPETITION SUMMARY
    # ========================================================

    summary = {
        "repetition":
            repetition,

        "questions":
            len(rows),

        "single_accuracy":
            accuracy(
                rows,
                "single_correct",
            ),

        "forest_accuracy":
            accuracy(
                rows,
                "forest_correct",
            ),

        "self_refine_accuracy":
            accuracy(
                rows,
                "self_refine_correct",
            ),

        "student_critic_accuracy":
            accuracy(
                rows,
                "student_critic_correct",
            ),

        "forest_corrections":
            correction_summary(
                "forest"
            ),

        "self_refine_corrections":
            correction_summary(
                "self_refine"
            ),

        "student_critic_corrections":
            correction_summary(
                "student_critic"
            ),

        "critic_quality":
            critic_metrics(
                critic_records
            ),

        "single_total_tokens":
            sum(
                row[
                    "single_tokens"
                ]
                for row
                in rows
            ),

        "forest_total_tokens":
            sum(
                row[
                    "forest_tokens"
                ]
                for row
                in rows
            ),

        "self_refine_total_tokens":
            sum(
                row[
                    "self_refine_tokens"
                ]
                for row
                in rows
            ),

        "student_critic_total_tokens":
            sum(
                row[
                    "student_critic_tokens"
                ]
                for row
                in rows
            ),

        "stage_wall_time_seconds":
            {
                "single":
                    single_time,

                "forest_extra":
                    forest_time,

                "self_refine_extra":
                    self_time,

                "student_critic_extra":
                    critic_time,
            },
    }

    return (
        rows,
        summary,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # IMPORTANT:
    # Initialize vLLM only inside main().
    # This avoids recursive initialization when multiprocessing
    # uses the spawn method.
    initialize_engine()

    print()
    print(
        "=" * 80
    )

    print(
        "CONTROLLED MULTI-AGENT REASONING EXPERIMENT"
    )

    print(
        "=" * 80
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Questions:",
        N_QUESTIONS,
    )

    print(
        "Repetitions:",
        N_REPETITIONS,
    )

    print(
        "Forest N:",
        FOREST_N,
    )

    print(
        "Self-Refine rounds:",
        REFINE_ROUNDS,
    )

    print(
        "Student-Critic rounds:",
        CRITIC_ROUNDS,
    )

    print(
        "Temperature:",
        TEMPERATURE,
    )

    print(
        "Max new tokens:",
        MAX_NEW_TOKENS,
    )

    print(
        "Max model length:",
        MAX_MODEL_LEN,
    )

    print(
        "Max active sequences:",
        MAX_NUM_SEQS,
    )

    print(
        "Results:",
        RESULTS_DIR,
    )

    print(
        "=" * 80
    )

    questions = (
        load_math_sample()
    )

    save_json(
        RESULTS_DIR
        / "sampled_questions.json",
        questions,
    )

    raw_results_path = (
        RESULTS_DIR
        / "raw_results.jsonl"
    )

    if (
        raw_results_path.exists()
    ):
        raw_results_path.unlink()

    summaries = []

    for repetition in range(
        1,
        N_REPETITIONS + 1,
    ):

        (
            rows,
            summary,
        ) = run_repetition(
            questions,
            repetition,
        )

        append_jsonl(
            raw_results_path,
            rows,
        )

        save_json(
            RESULTS_DIR
            /
            f"repetition_{repetition}_summary.json",
            summary,
        )

        summaries.append(
            summary
        )

        save_json(
            RESULTS_DIR
            / "summaries.json",
            summaries,
        )

        print()
        print(
            "-" * 80
        )

        print(
            f"SUMMARY REPETITION "
            f"{repetition}"
        )

        print(
            "-" * 80
        )

        print(
            f"Single: "
            f"{summary['single_accuracy']:.2%}"
        )

        print(
            f"Forest N={FOREST_N}: "
            f"{summary['forest_accuracy']:.2%}"
        )

        print(
            f"Self-Refine: "
            f"{summary['self_refine_accuracy']:.2%}"
        )

        print(
            f"Student-Critic: "
            f"{summary['student_critic_accuracy']:.2%}"
        )

        critic_quality = (
            summary[
                "critic_quality"
            ]
        )

        precision = (
            critic_quality[
                "error_detection_precision"
            ]
        )

        recall = (
            critic_quality[
                "error_detection_recall"
            ]
        )

        print(
            "Critic precision:",
            (
                f"{precision:.2%}"
                if precision
                is not None
                else "NA"
            ),
        )

        print(
            "Critic recall:",
            (
                f"{recall:.2%}"
                if recall
                is not None
                else "NA"
            ),
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    fields = [
        "single_accuracy",
        "forest_accuracy",
        "self_refine_accuracy",
        "student_critic_accuracy",
    ]

    final_summary = {}

    for field in fields:

        values = [
            summary[field]
            for summary
            in summaries
        ]

        final_summary[
            field
        ] = {
            "values":
                values,

            "mean":
                mean(values),

            "std_population":
                (
                    pstdev(values)
                    if len(values)
                    > 1
                    else 0.0
                ),
        }

    save_json(
        RESULTS_DIR
        / "final_summary.json",
        final_summary,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL MEAN RESULTS"
    )

    print(
        "=" * 80
    )

    for field in fields:

        print(
            f"{field:<30} "
            f"{final_summary[field]['mean']:.2%}"
        )

    print()

    print(
        "Results saved to:",
        RESULTS_DIR,
    )


# ============================================================
# SAFE MULTIPROCESSING ENTRY POINT
# ============================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()