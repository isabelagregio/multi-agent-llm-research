#!/usr/bin/env python3
"""
Experiment E — Cost-Aware Evaluation

Purpose
-------
Quantify the accuracy/computation trade-off of the strategies evaluated in
Experiment D and identify whether Adaptive Self-Refine lies on the Pareto
frontier.

This experiment is intentionally OFFLINE: it reuses Experiment D trajectories
and does not perform new LLM inference.

Inputs expected from Experiment D
---------------------------------
results/experiment_D_adaptive_stopping/
    raw_results.jsonl
    final_summary.json
    repetition_1_summary.json
    repetition_2_summary.json
    repetition_3_summary.json
    runtime_validation_rep_*.json   # optional

Main questions
--------------
1. How much token cost is required by each strategy?
2. Which strategies are Pareto-efficient in accuracy vs. token usage?
3. How much accuracy is gained/lost per unit of additional compute?
4. Does Adaptive Self-Refine preserve accuracy while reducing cost?
5. At a target accuracy, which strategy is the cheapest?
6. If true runtime-validation files exist, do token savings also translate
   into measured wall-time savings?

Statistical treatment
---------------------
The same 500 MATH questions are used across 3 stochastic repetitions.
Therefore, bootstrap confidence intervals are CLUSTERED by question_id:
a sampled question contributes all of its repetitions together.

Outputs
-------
results/experiment_E_cost_aware/
    strategy_summary.csv
    pairwise_adaptive_vs_fixed.csv
    pareto_frontier.csv
    target_accuracy_cost.csv
    final_summary.json
    quick_read.txt
    accuracy_vs_tokens.png
    correct_per_1k_tokens.png
    accuracy_vs_mean_rounds.png
    runtime_validation_summary.csv   # only if runtime files exist
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/lustre06/project/6087921/isa1909")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "experiment_D_adaptive_stopping"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment_E_cost_aware"

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261013

STRATEGIES = ["R0", "R1", "R2", "R3", "Adaptive"]


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Convert Experiment D rows to long format
# ---------------------------------------------------------------------

def to_long_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    One row per (question_id, repetition, strategy).

    The adaptive token count is the counterfactual token consumption from
    Experiment D: only stages that the adaptive policy would have executed.
    """
    out = []

    for row in raw_rows:
        base = {
            "question_id": row["question_id"],
            "repetition": row["repetition"],
            "level": row["level"],
        }

        for r in range(4):
            item = row["fixed"][str(r)]
            out.append({
                **base,
                "strategy": f"R{r}",
                "accuracy": 1.0 if item["correct"] else 0.0,
                "tokens": float(item["tokens"]),
                "rounds": float(r),
            })

        a = row["adaptive"]
        out.append({
            **base,
            "strategy": "Adaptive",
            "accuracy": 1.0 if a["correct"] else 0.0,
            "tokens": float(a["tokens"]),
            "rounds": float(a["stop_round"]),
        })

    return out


def group_by_strategy(long_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in long_rows:
        grouped[row["strategy"]].append(row)
    return grouped


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def summarize_strategy(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    acc = mean(r["accuracy"] for r in rows)
    tok = mean(r["tokens"] for r in rows)
    rounds = mean(r["rounds"] for r in rows)

    return {
        "accuracy": acc,
        "mean_tokens_per_question": tok,
        "mean_rounds": rounds,
        # Expected number of correct answers that can be produced per 1000 tokens
        # if the same average accuracy/cost is maintained.
        "correct_answers_per_1k_tokens": (acc * 1000.0 / tok) if tok else float("nan"),
    }


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w


# ---------------------------------------------------------------------
# Clustered bootstrap
# ---------------------------------------------------------------------

def cluster_bootstrap(
    long_rows: List[Dict[str, Any]],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Cluster by question_id so all repetitions of a sampled question stay
    together. Question IDs are sampled with replacement.
    """
    by_qid: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        by_qid[row["question_id"]].append(row)

    qids = sorted(by_qid.keys())
    rng = random.Random(seed)

    boot_strategy_metrics = {
        s: {
            "accuracy": [],
            "tokens": [],
            "rounds": [],
            "correct_per_1k": [],
        }
        for s in STRATEGIES
    }

    pairwise = {
        fixed: {
            "accuracy_diff_adaptive_minus_fixed": [],
            "token_diff_adaptive_minus_fixed": [],
            "token_savings_fraction": [],
        }
        for fixed in ["R0", "R1", "R2", "R3"]
    }

    for _ in range(n_boot):
        sampled_qids = [rng.choice(qids) for _ in qids]

        # Sampling with replacement means clusters may appear multiple times.
        sample_rows = []
        for sampled_index, qid in enumerate(sampled_qids):
            for row in by_qid[qid]:
                copy = dict(row)
                copy["_bootstrap_cluster_instance"] = sampled_index
                sample_rows.append(copy)

        grouped = group_by_strategy(sample_rows)
        metrics = {s: summarize_strategy(grouped[s]) for s in STRATEGIES}

        for s in STRATEGIES:
            boot_strategy_metrics[s]["accuracy"].append(metrics[s]["accuracy"])
            boot_strategy_metrics[s]["tokens"].append(metrics[s]["mean_tokens_per_question"])
            boot_strategy_metrics[s]["rounds"].append(metrics[s]["mean_rounds"])
            boot_strategy_metrics[s]["correct_per_1k"].append(
                metrics[s]["correct_answers_per_1k_tokens"]
            )

        a = metrics["Adaptive"]
        for fixed in ["R0", "R1", "R2", "R3"]:
            f = metrics[fixed]
            pairwise[fixed]["accuracy_diff_adaptive_minus_fixed"].append(
                a["accuracy"] - f["accuracy"]
            )
            pairwise[fixed]["token_diff_adaptive_minus_fixed"].append(
                a["mean_tokens_per_question"] - f["mean_tokens_per_question"]
            )
            pairwise[fixed]["token_savings_fraction"].append(
                (f["mean_tokens_per_question"] - a["mean_tokens_per_question"])
                / f["mean_tokens_per_question"]
            )

    return {
        "strategy_metrics": boot_strategy_metrics,
        "pairwise": pairwise,
    }


def ci95(values: List[float]) -> Tuple[float, float]:
    return percentile(values, 0.025), percentile(values, 0.975)


# ---------------------------------------------------------------------
# Pareto and target-accuracy analyses
# ---------------------------------------------------------------------

def pareto_frontier(strategy_metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Minimize tokens, maximize accuracy.

    Strategy X is dominated if another strategy has:
        tokens <= X.tokens
        accuracy >= X.accuracy
    and is strictly better in at least one dimension.
    """
    rows = []

    for s, m in strategy_metrics.items():
        dominated_by = []
        for t, tm in strategy_metrics.items():
            if t == s:
                continue
            no_more_cost = tm["mean_tokens_per_question"] <= m["mean_tokens_per_question"]
            no_less_acc = tm["accuracy"] >= m["accuracy"]
            strict = (
                tm["mean_tokens_per_question"] < m["mean_tokens_per_question"]
                or tm["accuracy"] > m["accuracy"]
            )
            if no_more_cost and no_less_acc and strict:
                dominated_by.append(t)

        rows.append({
            "strategy": s,
            "accuracy": m["accuracy"],
            "mean_tokens_per_question": m["mean_tokens_per_question"],
            "mean_rounds": m["mean_rounds"],
            "pareto_efficient": len(dominated_by) == 0,
            "dominated_by": ";".join(dominated_by),
        })

    rows.sort(key=lambda x: x["mean_tokens_per_question"])
    return rows


def target_accuracy_table(
    strategy_metrics: Dict[str, Dict[str, float]],
    step: float = 0.005,
) -> List[Dict[str, Any]]:
    """
    For target accuracies from R0 accuracy through the best observed accuracy,
    return the cheapest strategy satisfying the target.
    """
    min_acc = min(m["accuracy"] for m in strategy_metrics.values())
    max_acc = max(m["accuracy"] for m in strategy_metrics.values())

    start = math.floor(min_acc / step) * step
    end = math.ceil(max_acc / step) * step

    thresholds = []
    x = start
    while x <= end + 1e-12:
        thresholds.append(round(x, 6))
        x += step

    rows = []
    for target in thresholds:
        eligible = [
            (s, m)
            for s, m in strategy_metrics.items()
            if m["accuracy"] + 1e-12 >= target
        ]
        if not eligible:
            rows.append({
                "target_accuracy": target,
                "cheapest_strategy": "",
                "tokens_per_question": "",
                "observed_accuracy": "",
            })
            continue

        s, m = min(eligible, key=lambda x: x[1]["mean_tokens_per_question"])
        rows.append({
            "target_accuracy": target,
            "cheapest_strategy": s,
            "tokens_per_question": m["mean_tokens_per_question"],
            "observed_accuracy": m["accuracy"],
        })

    return rows


# ---------------------------------------------------------------------
# Optional real runtime validation from Experiment D
# ---------------------------------------------------------------------

def load_runtime_validation(input_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("runtime_validation_rep_*.json")):
        data = load_json(path)
        rows.append({
            "repetition": data["repetition"],
            "strategy": "Adaptive",
            "accuracy": data["accuracy"],
            "mean_rounds": data["mean_rounds"],
            "mean_tokens_per_question": data["mean_tokens_per_question"],
            "actual_wall_time_seconds": data["actual_wall_time_seconds"],
            "wall_time_per_question_seconds": (
                data["actual_wall_time_seconds"] / data["questions"]
            ),
        })
    return rows


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_accuracy_vs_tokens(
    strategy_metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    ordered = sorted(
        strategy_metrics.items(),
        key=lambda x: x[1]["mean_tokens_per_question"],
    )

    plt.figure(figsize=(7, 4.6))
    for strategy, m in ordered:
        plt.scatter(
            m["mean_tokens_per_question"],
            100 * m["accuracy"],
            s=70,
        )
        plt.annotate(
            strategy,
            (m["mean_tokens_per_question"], 100 * m["accuracy"]),
            xytext=(6, 5),
            textcoords="offset points",
        )

    plt.xlabel("Mean tokens per question")
    plt.ylabel("Accuracy (%)")
    plt.title("Experiment E — Accuracy vs. Token Cost")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correct_per_1k(
    strategy_metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    strategies = STRATEGIES
    values = [
        strategy_metrics[s]["correct_answers_per_1k_tokens"]
        for s in strategies
    ]

    plt.figure(figsize=(7, 4.6))
    bars = plt.bar(strategies, values)

    plt.ylabel("Expected correct answers per 1k tokens")
    plt.title("Experiment E — Token Efficiency")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_rounds(
    strategy_metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    ordered = sorted(
        strategy_metrics.items(),
        key=lambda x: x[1]["mean_rounds"],
    )

    plt.figure(figsize=(7, 4.6))
    for strategy, m in ordered:
        plt.scatter(
            m["mean_rounds"],
            100 * m["accuracy"],
            s=70,
        )
        plt.annotate(
            strategy,
            (m["mean_rounds"], 100 * m["accuracy"]),
            xytext=(6, 5),
            textcoords="offset points",
        )

    plt.xlabel("Mean interaction rounds")
    plt.ylabel("Accuracy (%)")
    plt.title("Experiment E — Accuracy vs. Interaction Depth")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def build_strategy_summary_rows(
    point_metrics: Dict[str, Dict[str, float]],
    boot: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for s in STRATEGIES:
        m = point_metrics[s]
        b = boot["strategy_metrics"][s]

        acc_lo, acc_hi = ci95(b["accuracy"])
        tok_lo, tok_hi = ci95(b["tokens"])
        rounds_lo, rounds_hi = ci95(b["rounds"])
        eff_lo, eff_hi = ci95(b["correct_per_1k"])

        rows.append({
            "strategy": s,
            "accuracy": m["accuracy"],
            "accuracy_ci95_low": acc_lo,
            "accuracy_ci95_high": acc_hi,
            "mean_tokens_per_question": m["mean_tokens_per_question"],
            "tokens_ci95_low": tok_lo,
            "tokens_ci95_high": tok_hi,
            "mean_rounds": m["mean_rounds"],
            "rounds_ci95_low": rounds_lo,
            "rounds_ci95_high": rounds_hi,
            "correct_answers_per_1k_tokens": m["correct_answers_per_1k_tokens"],
            "correct_per_1k_ci95_low": eff_lo,
            "correct_per_1k_ci95_high": eff_hi,
        })
    return rows


def build_pairwise_rows(
    point_metrics: Dict[str, Dict[str, float]],
    boot: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    a = point_metrics["Adaptive"]

    for fixed in ["R0", "R1", "R2", "R3"]:
        f = point_metrics[fixed]
        b = boot["pairwise"][fixed]

        acc_diff = a["accuracy"] - f["accuracy"]
        tok_diff = a["mean_tokens_per_question"] - f["mean_tokens_per_question"]
        savings = (
            f["mean_tokens_per_question"] - a["mean_tokens_per_question"]
        ) / f["mean_tokens_per_question"]

        acc_lo, acc_hi = ci95(b["accuracy_diff_adaptive_minus_fixed"])
        tok_lo, tok_hi = ci95(b["token_diff_adaptive_minus_fixed"])
        sav_lo, sav_hi = ci95(b["token_savings_fraction"])

        rows.append({
            "comparison": f"Adaptive - {fixed}",
            "accuracy_difference": acc_diff,
            "accuracy_diff_ci95_low": acc_lo,
            "accuracy_diff_ci95_high": acc_hi,
            "token_difference": tok_diff,
            "token_diff_ci95_low": tok_lo,
            "token_diff_ci95_high": tok_hi,
            "token_savings_fraction": savings,
            "token_savings_ci95_low": sav_lo,
            "token_savings_ci95_high": sav_hi,
        })

    return rows


def write_quick_read(
    path: Path,
    strategy_rows: List[Dict[str, Any]],
    pairwise_rows: List[Dict[str, Any]],
    pareto_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
) -> None:
    lines = [
        "EXPERIMENT E — COST-AWARE EVALUATION",
        "",
        "Strategy summary:",
    ]

    for row in strategy_rows:
        lines.append(
            f"  {row['strategy']}: "
            f"accuracy={100*row['accuracy']:.2f}% "
            f"[95% CI {100*row['accuracy_ci95_low']:.2f}, "
            f"{100*row['accuracy_ci95_high']:.2f}] | "
            f"tokens/q={row['mean_tokens_per_question']:.1f} | "
            f"mean rounds={row['mean_rounds']:.3f} | "
            f"correct/1k tokens={row['correct_answers_per_1k_tokens']:.3f}"
        )

    lines.extend(["", "Adaptive vs fixed strategies:"])
    for row in pairwise_rows:
        lines.append(
            f"  {row['comparison']}: "
            f"accuracy diff={100*row['accuracy_difference']:+.2f} pp "
            f"[95% CI {100*row['accuracy_diff_ci95_low']:+.2f}, "
            f"{100*row['accuracy_diff_ci95_high']:+.2f}] | "
            f"token savings={100*row['token_savings_fraction']:+.2f}% "
            f"[95% CI {100*row['token_savings_ci95_low']:+.2f}, "
            f"{100*row['token_savings_ci95_high']:+.2f}]"
        )

    lines.extend(["", "Pareto-efficient strategies:"])
    for row in pareto_rows:
        if row["pareto_efficient"]:
            lines.append(
                f"  {row['strategy']} "
                f"(accuracy={100*row['accuracy']:.2f}%, "
                f"tokens/q={row['mean_tokens_per_question']:.1f})"
            )

    if runtime_rows:
        lines.extend(["", "Actual adaptive runtime validation:"])
        for row in runtime_rows:
            lines.append(
                f"  rep {row['repetition']}: "
                f"accuracy={100*row['accuracy']:.2f}% | "
                f"tokens/q={row['mean_tokens_per_question']:.1f} | "
                f"wall={row['actual_wall_time_seconds']:.1f}s"
            )
    else:
        lines.extend([
            "",
            "Actual runtime validation:",
            "  Not available. No runtime_validation_rep_*.json files were found.",
            "  Token results remain valid; real latency savings have not yet been measured.",
        ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Experiment D result directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Experiment E output directory.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = input_dir / "raw_results.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Experiment E requires Experiment D raw_results.jsonl."
        )

    raw_rows = load_jsonl(raw_path)
    if not raw_rows:
        raise RuntimeError("Experiment D raw_results.jsonl is empty.")

    long_rows = to_long_rows(raw_rows)
    grouped = group_by_strategy(long_rows)

    missing = [s for s in STRATEGIES if s not in grouped]
    if missing:
        raise RuntimeError(f"Missing strategies in Experiment D results: {missing}")

    point_metrics = {
        s: summarize_strategy(grouped[s])
        for s in STRATEGIES
    }

    print(f"Loaded {len(raw_rows)} Experiment D question/repetition rows.")
    print("Running clustered bootstrap...")
    boot = cluster_bootstrap(
        long_rows,
        n_boot=args.bootstrap_samples,
        seed=BOOTSTRAP_SEED,
    )

    strategy_rows = build_strategy_summary_rows(point_metrics, boot)
    pairwise_rows = build_pairwise_rows(point_metrics, boot)
    pareto_rows = pareto_frontier(point_metrics)
    target_rows = target_accuracy_table(point_metrics)

    runtime_rows = load_runtime_validation(input_dir)

    write_csv(output_dir / "strategy_summary.csv", strategy_rows)
    write_csv(output_dir / "pairwise_adaptive_vs_fixed.csv", pairwise_rows)
    write_csv(output_dir / "pareto_frontier.csv", pareto_rows)
    write_csv(output_dir / "target_accuracy_cost.csv", target_rows)

    if runtime_rows:
        write_csv(output_dir / "runtime_validation_summary.csv", runtime_rows)

    plot_accuracy_vs_tokens(
        point_metrics,
        output_dir / "accuracy_vs_tokens.png",
    )
    plot_correct_per_1k(
        point_metrics,
        output_dir / "correct_per_1k_tokens.png",
    )
    plot_accuracy_vs_rounds(
        point_metrics,
        output_dir / "accuracy_vs_mean_rounds.png",
    )

    final = {
        "experiment": "E - Cost-Aware Evaluation",
        "source_experiment": "D - Adaptive Self-Refine",
        "rows_from_experiment_D": len(raw_rows),
        "unique_questions": len(set(r["question_id"] for r in raw_rows)),
        "repetitions": len(set(r["repetition"] for r in raw_rows)),
        "bootstrap": {
            "method": "clustered bootstrap by question_id",
            "samples": args.bootstrap_samples,
            "seed": BOOTSTRAP_SEED,
        },
        "strategy_summary": strategy_rows,
        "adaptive_pairwise_comparisons": pairwise_rows,
        "pareto_frontier": pareto_rows,
        "target_accuracy_cost": target_rows,
        "runtime_validation_available": bool(runtime_rows),
        "runtime_validation": runtime_rows,
        "important_note": (
            "Adaptive token counts come from the paired Experiment D full-R3 "
            "trajectories and represent counterfactual consumed tokens under "
            "early stopping. Actual latency is reported only when Experiment D "
            "runtime_validation_rep_*.json files are present."
        ),
    }

    write_json(output_dir / "final_summary.json", final)
    write_quick_read(
        output_dir / "quick_read.txt",
        strategy_rows,
        pairwise_rows,
        pareto_rows,
        runtime_rows,
    )

    print("\nExperiment E complete.")
    print(f"Results: {output_dir}")
    print()
    print((output_dir / "quick_read.txt").read_text())


if __name__ == "__main__":
    main()
