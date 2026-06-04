"""Export every S3 lighteval eval into one wide CSV (joel-board parity).

Reproduces the joel-board dashboard's English ``prob`` ``agg_score_metrics``
export entirely from S3: one row per ``(runname, seed, step)`` holding the twelve
English lighteval tasks (``prob_norm_token``), ``agg_score_micro``, the six
task-type category aggregates, and ``agg_score_macro``.

Runs and steps are discovered directly under ``S3_EVALS_PATH``; this command never
reads ``training_metadata.json`` or any other local file, so the proportion-sweep
(``mix-0.X-...``) and variance-check (``...-seed-<N>``) families are picked up
automatically alongside the leaderboard runs.

Run with: ``export-benchmark-csv`` (optionally ``--runs-regex`` / ``--output``).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from fsspec.core import url_to_fs
from joblib import Parallel, delayed
from tqdm import tqdm

from finephrase.benchmark import (
    BENCHMARK_CATEGORIES,
    BENCHMARK_METRIC,
    BENCHMARK_TASKS,
    S3_EVALS_PATH,
    parse_run_name_seed,
    short_task_name,
)
from finephrase.utils import PROJECT_PATH

logger = logging.getLogger(__name__)

# Category column order in the dashboard export: the order each task type first
# appears in the English agg_score list. ``agg_score_macro`` follows these.
CATEGORY_ORDER: list[str] = ["RC", "GK", "NLU", "MATH", "TABLE", "RES"]

# Task columns: non-average tasks alphabetically, then ":_average" tasks, which
# the dashboard appends last as recomputed average columns.
_NON_AVG_TASKS = sorted(t for t in BENCHMARK_TASKS if ":_average" not in t)
_AVG_TASKS = sorted(t for t in BENCHMARK_TASKS if ":_average" in t)
TASK_COLUMNS: list[str] = [f"{t}/{BENCHMARK_METRIC}" for t in _NON_AVG_TASKS + _AVG_TASKS]

# Final, fixed CSV column order (23 columns), identical to the dashboard export.
CSV_COLUMNS: list[str] = (
    ["runname", "seed", "steps", "agg_score_micro"]
    + TASK_COLUMNS
    + [f"agg_score_{category}" for category in CATEGORY_ORDER]
    + ["agg_score_macro"]
)

DEFAULT_OUTPUT = Path(PROJECT_PATH) / "benchmark-results.csv"
_RESULTS_ROOT = S3_EVALS_PATH.replace("s3://", "")
_TIMESTAMP_RE = re.compile(r"results_(.+)\.json$")

# Decay-resumed runs (e.g. fw_edu_lq-decay-fw_edu_hq) all carry this substring and are
# excluded by default; pass --include-decay to keep them.
DECAY_MARKER = "-decay-"

# Row value type: strings/ints for the identifier columns, floats for the scores.
Row = dict[str, str | int | float]


def build_row(runname: str, seed: int, step: int, scores: dict[str, float]) -> Row:
    """Assemble one CSV row: raw task scores plus fill-0 aggregate scores.

    Missing tasks count as ``0.0`` (the dashboard fills NaNs with 0 before
    averaging), so ``agg_score_micro`` always divides by the full twelve tasks and
    each category average divides by its full member count.
    """

    def value(short: str) -> float:
        return scores[short] if short in scores else 0.0

    row: Row = {"runname": runname, "seed": seed, "steps": step}
    for task in BENCHMARK_TASKS:
        row[f"{task}/{BENCHMARK_METRIC}"] = value(short_task_name(task))

    row["agg_score_micro"] = sum(row[col] for col in TASK_COLUMNS) / len(TASK_COLUMNS)

    category_means: list[float] = []
    for category in CATEGORY_ORDER:
        members = BENCHMARK_CATEGORIES[category]
        category_mean = sum(value(member) for member in members) / len(members)
        row[f"agg_score_{category}"] = category_mean
        category_means.append(category_mean)
    row["agg_score_macro"] = sum(category_means) / len(category_means)
    return row


def fetch_run_rows(run_folder: str, fs: object) -> list[Row]:
    """Build one CSV row per evaluated step for a single S3 run folder.

    Within a step, the newest ``results_*.json`` wins per task (ascending
    timestamp, later files overwrite earlier ones), mirroring the dashboard's
    per-task merge. Steps with none of the twelve English tasks are skipped.
    """
    runname, seed = parse_run_name_seed(run_folder)
    try:
        result_files = fs.glob(f"{_RESULTS_ROOT}/{run_folder}/*/results_*.json")
    except FileNotFoundError:
        return []

    # step -> [(timestamp_str, file_path), ...]
    files_by_step: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for path in result_files:
        step = int(path.split("/")[-2])
        timestamp = _TIMESTAMP_RE.search(path.split("/")[-1]).group(1)
        files_by_step[step].append((timestamp, path))

    rows: list[Row] = []
    for step, timestamp_paths in files_by_step.items():
        scores: dict[str, float] = {}
        for _, path in sorted(timestamp_paths):  # ascending timestamp: newest wins
            with fs.open(path, "r") as f:
                results = json.load(f)["results"]
            for task in BENCHMARK_TASKS:
                if task in results and BENCHMARK_METRIC in results[task]:
                    scores[short_task_name(task)] = results[task][BENCHMARK_METRIC]
        if scores:
            rows.append(build_row(runname, seed, step, scores))
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Export all S3 lighteval evals to a wide CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument(
        "--runs-regex", type=str, default=None, help="Only include S3 run folders fully matching this regex."
    )
    parser.add_argument(
        "--include-decay", action="store_true", help=f"Include decay-resumed runs (those with '{DECAY_MARKER}')."
    )
    parser.add_argument("--n-jobs", type=int, default=64, help="Parallel S3 reader threads.")
    args = parser.parse_args()

    fs, _ = url_to_fs(S3_EVALS_PATH)
    run_folders = sorted(path.split("/")[-1] for path in fs.ls(_RESULTS_ROOT, detail=False))
    if args.runs_regex:
        pattern = re.compile(args.runs_regex)
        run_folders = [run for run in run_folders if pattern.fullmatch(run)]
    if not args.include_decay:
        kept = [run for run in run_folders if DECAY_MARKER not in run]
        logger.info("Excluding %d decay run(s) (pass --include-decay to keep)", len(run_folders) - len(kept))
        run_folders = kept
    logger.info("Found %d run folders under %s", len(run_folders), S3_EVALS_PATH)

    # I/O-bound S3 reads: fan out over threads, stream results for a live progress bar.
    results = Parallel(n_jobs=args.n_jobs, prefer="threads", return_as="generator")(
        delayed(fetch_run_rows)(run_folder, fs) for run_folder in run_folders
    )
    rows: list[Row] = [
        row for run_rows in tqdm(results, total=len(run_folders), desc="Fetching evals from S3") for row in run_rows
    ]
    rows.sort(key=lambda r: (r["runname"], r["seed"], r["steps"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_series = len({(r["runname"], r["seed"]) for r in rows})
    logger.info("Wrote %d rows (%d run/seed series) to %s", len(rows), n_series, args.output)


if __name__ == "__main__":
    main()
