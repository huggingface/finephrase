"""Shared lighteval-benchmark constants and S3 result fetcher.

Used by both the rephrasing-metadata collector (``collect_rephrasing_metadata``)
and the training-metadata collector (``collect_training_metadata``) so the two
output files have an identical ``results`` schema and any change to the metric or
task list propagates everywhere.
"""

from __future__ import annotations

import json

S3_EVALS_PATH = "s3://finephrase/experiments/evals-test/results"
FINAL_CHECKPOINT = "10000"

# English benchmark tasks (from joel-board dashboard agg_score_metrics.py).
# In "prob" mode (default), the dashboard transforms all metrics to prob_norm_token.
BENCHMARK_TASKS: list[str] = [
    "lighteval|squad_v2|3",
    "lighteval|arc_cf:easy|3",
    "lighteval|hellaswag_cf|3",
    "lighteval|mmlu_redux_cf:_average|3",
    "lighteval|gsm8k|3",
    "lighteval|drop|3",
    "lighteval|wikitablequestions|3",
    "lighteval|treb_qa|3",
    "lighteval|winogrande_cf|3",
    "lighteval|piqa_cf|3",
    "lighteval|openbookqa_cf|3",
    "lighteval|xcsqa_cf|3",
]
BENCHMARK_METRIC = "prob_norm_token"

# Category groupings for aggregate scores (from joel-board task_type_mapping.py).
BENCHMARK_CATEGORIES: dict[str, list[str]] = {
    "GK": ["arc_cf:easy", "mmlu_redux_cf:_average"],
    "RC": ["squad_v2", "drop"],
    "RES": ["openbookqa_cf", "piqa_cf", "xcsqa_cf"],
    "NLU": ["hellaswag_cf", "winogrande_cf"],
    "MATH": ["gsm8k"],
    "TABLE": ["wikitablequestions", "treb_qa"],
}


def short_task_name(task_key: str) -> str:
    """Strip 'lighteval|' prefix and '|3' suffix from a task key."""
    return task_key.removeprefix("lighteval|").removesuffix("|3")


def parse_run_name_seed(run_folder: str) -> tuple[str, int]:
    """Split an S3 run folder into ``(runname, seed)`` like the joel-board dashboard.

    Seed-sweep runs carry a trailing ``-seed-<N>`` (e.g.
    ``fw_edu_hq--data-seed=1-seed=1-seed-101``): everything before it is the run
    name and ``<N>`` is the seed. Runs without that suffix default to seed ``42``
    (the dashboard's convention for single-seed runs).
    """
    if "-seed-" not in run_folder:
        return run_folder, 42
    name, _, seed = run_folder.rpartition("-seed-")
    return name, int(seed)


def fetch_benchmark_results(training_run: str, fs: object) -> dict[str, float] | None:
    """Fetch benchmark evaluation results from S3 for a training run.

    Reads the latest ``results_*.json`` at ``FINAL_CHECKPOINT``, extracts each
    task's ``BENCHMARK_METRIC``, and adds category averages (``agg_score_<cat>``),
    a macro average (``agg_score_macro``), and a micro average (``agg_score_micro``).
    Returns ``None`` if no results are found.
    """
    results_path = f"{S3_EVALS_PATH}/{training_run}/{FINAL_CHECKPOINT}".replace("s3://", "")
    try:
        result_files = fs.glob(f"{results_path}/results_*.json")
    except FileNotFoundError:
        return None
    if not result_files:
        return None

    # Latest file wins (lighteval names them with an iso timestamp).
    result_files.sort()
    with fs.open(result_files[-1], "r") as f:
        data = json.load(f)

    raw_results = data["results"]
    scores: dict[str, float] = {}
    for task_key in BENCHMARK_TASKS:
        if task_key in raw_results and BENCHMARK_METRIC in raw_results[task_key]:
            scores[short_task_name(task_key)] = raw_results[task_key][BENCHMARK_METRIC]

    if not scores:
        return None

    # Per-category mean of present member scores.
    category_scores: list[float] = []
    for category, members in BENCHMARK_CATEGORIES.items():
        member_scores = [scores[m] for m in members if m in scores]
        if member_scores:
            cat_avg = sum(member_scores) / len(member_scores)
            scores[f"agg_score_{category}"] = cat_avg
            category_scores.append(cat_avg)

    if category_scores:
        scores["agg_score_macro"] = sum(category_scores) / len(category_scores)

    # Micro = mean of individual benchmarks (i.e. exclude derived agg_score_* keys).
    individual = [v for k, v in scores.items() if not k.startswith("agg_score_")]
    if individual:
        scores["agg_score_micro"] = sum(individual) / len(individual)

    return scores
