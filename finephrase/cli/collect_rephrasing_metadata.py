"""Collect metadata from rephrasing runs into a structured JSON file.

Walks beyondweb/, format/, nemotron/, rewire/ folders under the rephrasing logs dir,
filters to runs with >= 90 completions, loads or aggregates stats, and extracts
token counts, quality scores, GPU time, model, source dataset, prompt info,
and downstream benchmark evaluation results from S3.

Usage:
    python analysis/collect_rephrasing_metadata.py

Output is written to rephrasing_metadata.json in the project root.
"""

import json
import logging
import re
from pathlib import Path

from fsspec.core import url_to_fs
from tqdm import tqdm

from datatrove.utils.stats import PipelineStats

logger = logging.getLogger(__name__)

BASE_PATH = Path("/fsx/joel_niklaus/logs/finephrase/experiments/rephrasing")
CATEGORIES = ["beyondweb", "format", "nemotron", "rewire"]
MIN_COMPLETIONS = 90
S3_EVALS_PATH = "s3://finephrase/experiments/evals-test/results"
FINAL_CHECKPOINT = "10000"

# Maps source dataset names (from executor.json) to training mix base names
SOURCE_DATASET_SHORT_NAMES: dict[str, str] = {
    "fineweb-edu-hq-20BT": "fw_edu_hq",
    "fineweb-edu-lq-20BT": "fw_edu_lq",
    "dclm-37BT": "dclm",
    "cosmopedia-25BT": "cosmopedia",
}

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

# Category groupings for aggregate scores (from joel-board task_type_mapping.py)
BENCHMARK_CATEGORIES: dict[str, list[str]] = {
    "GK": ["arc_cf:easy", "mmlu_redux_cf:_average"],
    "RC": ["squad_v2", "drop"],
    "RES": ["openbookqa_cf", "piqa_cf", "xcsqa_cf"],
    "NLU": ["hellaswag_cf", "winogrande_cf"],
    "MATH": ["gsm8k"],
    "TABLE": ["wikitablequestions", "treb_qa"],
}

# Known prompt names per category, sorted longest-first to avoid prefix collisions
KNOWN_PROMPTS: dict[str, list[str]] = {
    "beyondweb": ["continue", "summarize"],
    "format": ["article", "commentary", "discussion", "faq", "tutorial", "table", "math"],
    "nemotron": [
        "diverse_qa_pairs",
        "extract_knowledge",
        "knowledge_list",
        "wikipedia_style_rephrasing",
        "distill",
    ],
    "rewire": ["guided_rewrite_original", "guided_rewrite_improved"],
}


def has_enough_completions(run_dir: Path) -> bool:
    """Check whether the run has at least MIN_COMPLETIONS completion files."""
    completions_dir = run_dir / "completions"
    if not completions_dir.exists():
        return False
    return len(list(completions_dir.iterdir())) >= MIN_COMPLETIONS


def load_or_aggregate_stats(run_dir: Path) -> list[dict] | None:
    """Load stats.json if it exists, otherwise aggregate per-rank stats files.

    Saves the aggregated result back to stats.json for future use.
    """
    stats_json_path = run_dir / "stats.json"

    if stats_json_path.exists():
        with open(stats_json_path) as f:
            return json.load(f)

    # Fallback: aggregate individual rank stats
    stats_dir = run_dir / "stats"
    if not stats_dir.exists():
        return None

    stat_files = sorted(stats_dir.glob("*.json"))
    if not stat_files:
        return None

    # Aggregate per-rank stats, skipping files with incompatible stage names
    merged = PipelineStats()
    skipped = 0
    for stat_file in tqdm(stat_files, desc=f"Aggregating {run_dir.name}", leave=False):
        with open(stat_file) as f:
            rank_stats = PipelineStats.from_json(json.load(f))
        try:
            merged = merged + rank_stats
        except AssertionError:
            skipped += 1
            continue
    if skipped:
        logger.info(f"Skipped {skipped}/{len(stat_files)} incompatible stats files for {run_dir.name}")

    if not merged.stats:
        return None

    # Persist aggregated stats for future use
    with open(stats_json_path, "w") as f:
        merged.save_to_disk(f)
    logger.info(f"Saved aggregated stats to {stats_json_path}")

    with open(stats_json_path) as f:
        return json.load(f)


def extract_model_and_dataset(run_dir: Path) -> tuple[str, str]:
    """Extract model name and source dataset from executor.json."""
    with open(run_dir / "executor.json") as f:
        executor = json.load(f)

    model = executor["pipeline"][1]["config"]["model_name_or_path"]

    # Parse DataFolder string: "DataFolder(path='finephrase/.../fineweb-edu-hq-20BT', fs=<...>)"
    data_folder_str = executor["pipeline"][0]["data_folder"]
    match = re.search(r"path='([^']+)'", data_folder_str)
    source_dataset = match.group(1).split("/")[-1] if match else "unknown"

    return model, source_dataset


def derive_prompt(category: str, run_name: str) -> str:
    """Derive the prompt file path from category and run folder name.

    Uses longest-prefix matching against known prompt names so that e.g.
    'guided_rewrite_original' matches before 'guided_rewrite_improved'.
    """
    prompts = sorted(KNOWN_PROMPTS[category], key=len, reverse=True)
    for prompt in prompts:
        if run_name.startswith(prompt + "-") or run_name == prompt:
            return f"{category}/{prompt}.md"
    logger.warning(f"Could not derive prompt for {category}/{run_name}, using folder name")
    return f"{category}/{run_name}.md"


def _format_count(n: float) -> str:
    """Format a large number with a human-readable suffix (e.g. 15.3B, 420.1M)."""
    for suffix, threshold in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= threshold:
            return f"{n / threshold:.1f}{suffix}"
    return str(int(n))


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    parts: list[str] = []
    remaining = int(seconds)

    for unit, divisor in [("years", 365 * 86400), ("months", 30 * 86400), ("days", 86400), ("hours", 3600), ("minutes", 60)]:
        if remaining >= divisor:
            count, remaining = divmod(remaining, divisor)
            parts.append(f"{count} {unit}" if count > 1 else f"{count} {unit.rstrip('s')}")

    # Always include seconds with fractional part
    frac_seconds = seconds - int(seconds)
    parts.append(f"{remaining + frac_seconds:.2f} seconds")

    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# GPU multiplier: larger models use tensor parallelism across multiple GPUs
_GPU_MULTIPLIER: dict[str, int] = {"12b": 2, "27b": 4}


def _gpu_multiplier_for_model(model_name: str) -> int:
    """Return the GPU multiplier based on model size (12b -> 2, 27b -> 4, else 1)."""
    name_lower = model_name.lower()
    for size_tag, multiplier in _GPU_MULTIPLIER.items():
        if size_tag in name_lower:
            return multiplier
    return 1


def _short_task_name(task_key: str) -> str:
    """Strip 'lighteval|' prefix and '|3' suffix from a task key."""
    return task_key.removeprefix("lighteval|").removesuffix("|3")


def derive_training_run_name(run_name: str, source_dataset: str) -> str | None:
    """Derive the training experiment name from a rephrasing run folder name and source dataset.

    Returns None if the source dataset is not in SOURCE_DATASET_SHORT_NAMES.
    E.g. run_name='faq-1b-hq', source_dataset='fineweb-edu-hq-20BT' -> 'mix-fw_edu_hq-faq_1b_hq'
    """
    base = SOURCE_DATASET_SHORT_NAMES.get(source_dataset)
    if base is None:
        return None
    tokenized_name = run_name.replace("-", "_")
    return f"mix-{base}-{tokenized_name}"


def fetch_benchmark_results(training_run: str, fs: object) -> dict[str, float] | None:
    """Fetch benchmark evaluation results from S3 for a training run.

    Reads the latest results JSON at step 10000, extracts scores using the correct
    metric per task, and computes category and overall aggregate scores.
    Returns None if no results are found.
    """
    results_path = f"{S3_EVALS_PATH}/{training_run}/{FINAL_CHECKPOINT}".replace("s3://", "")
    try:
        result_files = fs.glob(f"{results_path}/results_*.json")
    except FileNotFoundError:
        return None
    if not result_files:
        return None

    # Use the latest results file
    result_files.sort()
    with fs.open(result_files[-1], "r") as f:
        data = json.load(f)

    raw_results = data["results"]
    scores: dict[str, float] = {}

    # Extract individual benchmark scores (all use prob_norm_token in prob mode)
    for task_key in BENCHMARK_TASKS:
        if task_key in raw_results and BENCHMARK_METRIC in raw_results[task_key]:
            short_name = _short_task_name(task_key)
            scores[short_name] = raw_results[task_key][BENCHMARK_METRIC]

    if not scores:
        return None

    # Compute category aggregate scores
    category_scores: list[float] = []
    for category, members in BENCHMARK_CATEGORIES.items():
        member_scores = [scores[m] for m in members if m in scores]
        if member_scores:
            cat_avg = sum(member_scores) / len(member_scores)
            scores[f"agg_score_{category}"] = cat_avg
            category_scores.append(cat_avg)

    # Macro average: mean of category averages
    if category_scores:
        scores["agg_score_macro"] = sum(category_scores) / len(category_scores)

    # Micro average: mean of all individual benchmark scores (excluding aggregates)
    individual = [v for k, v in scores.items() if not k.startswith("agg_score_")]
    if individual:
        scores["agg_score_micro"] = sum(individual) / len(individual)

    return scores


def _stat_value(stats: dict, key: str, field: str) -> float:
    """Extract a field from a stat entry that may be a dict or scalar."""
    value = stats[key]
    if isinstance(value, dict):
        return value[field]
    return value


def _find_stage(stats: list[dict], name_substring: str) -> dict | None:
    """Find the first pipeline stage whose name contains the given substring."""
    for stage in stats:
        if name_substring in stage["name"]:
            return stage
    return None


def extract_metrics(stats: list[dict], model_name: str) -> dict[str, float | str]:
    """Extract token counts, GPU time, and quality scores from pipeline stats.

    Token counts come from the Quality Score stage (input_token_count / output_token_count),
    GPU time comes from the Inference stage's time_stats, multiplied by the number of GPUs
    used for tensor parallelism (2x for 12b models, 4x for 27b models).
    """
    metrics: dict[str, float | str] = {}
    inference = _find_stage(stats, "Inference")
    reader = _find_stage(stats, "READER")

    # Failed requests and num_documents first
    if inference:
        metrics["failed_requests"] = _stat_value(inference["stats"], "failed_requests", "total")
    if reader:
        metrics["num_documents"] = _stat_value(reader["stats"], "documents", "total")

    # GPU time from the inference stage, adjusted for tensor parallelism
    gpu_time_seconds: float | None = None
    if inference:
        multiplier = _gpu_multiplier_for_model(model_name)
        gpu_time_seconds = inference["time_stats"]["total"] * multiplier
        metrics["gpu_time_seconds"] = gpu_time_seconds
        metrics["gpu_time_human"] = _format_duration(gpu_time_seconds)

    # Token counts and quality scores from the quality/education stats logger stage
    quality = _find_stage(stats, "Quality Score") or _find_stage(stats, "Education Score")
    if quality:
        s = quality["stats"]

        # Output throughput per GPU (placed after gpu_time_human in output)
        output_tokens_total = _stat_value(s, "output_token_count", "total")
        if gpu_time_seconds and output_tokens_total:
            metrics["output_tps_per_gpu"] = round(output_tokens_total / gpu_time_seconds, 1)

        # Total token counts
        input_tokens = _stat_value(s, "input_token_count", "total")
        metrics["input_tokens"] = input_tokens
        metrics["input_tokens_human"] = _format_count(input_tokens)
        metrics["output_tokens"] = output_tokens_total
        metrics["output_tokens_human"] = _format_count(output_tokens_total)

        # Mean token counts
        metrics["input_token_count_mean"] = _stat_value(s, "input_token_count", "mean")
        metrics["output_token_count_mean"] = _stat_value(s, "output_token_count", "mean")
        metrics["token_reduction_mean"] = _stat_value(s, "token_reduction", "mean")

        # Compression ratio: output / input tokens
        if input_tokens and output_tokens_total:
            metrics["compression_ratio"] = round(output_tokens_total / input_tokens, 4)

        # Quality scores
        metrics["input_edu_score"] = _stat_value(s, "input_edu_score", "mean")
        metrics["output_edu_score"] = _stat_value(s, "output_edu_score", "mean")
        metrics["edu_score_difference"] = _stat_value(s, "edu_score_difference", "mean")
        metrics["edu_score_improvement"] = _stat_value(s, "edu_score_improvement", "mean")

        # DCLM scores (not present in older runs using Education Score stage)
        if "input_dclm_score" in s:
            metrics["input_dclm_score"] = _stat_value(s, "input_dclm_score", "mean")
            metrics["output_dclm_score"] = _stat_value(s, "output_dclm_score", "mean")
            metrics["dclm_score_difference"] = _stat_value(s, "dclm_score_difference", "mean")
            metrics["dclm_score_improvement"] = _stat_value(s, "dclm_score_improvement", "mean")

    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    fs, _ = url_to_fs(S3_EVALS_PATH)
    results: list[dict] = []

    for category in CATEGORIES:
        category_dir = BASE_PATH / category
        if not category_dir.exists():
            logger.warning(f"Category directory not found: {category_dir}")
            continue

        run_dirs = sorted(d for d in category_dir.iterdir() if d.is_dir())

        for run_dir in tqdm(run_dirs, desc=f"Processing {category}"):
            if not has_enough_completions(run_dir):
                logger.info(f"Skipping {category}/{run_dir.name}: not enough completions")
                continue

            try:
                model, source_dataset = extract_model_and_dataset(run_dir)
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Could not read executor.json for {category}/{run_dir.name}: {e}"
                )
                continue

            stats = load_or_aggregate_stats(run_dir)
            if stats is None:
                logger.warning(f"No stats found for {category}/{run_dir.name}")
                continue

            metrics = extract_metrics(stats, model)
            prompt = derive_prompt(category, run_dir.name)

            record: dict = {
                "run": f"{category}/{run_dir.name}",
                "model": model,
                "source_dataset": source_dataset,
                "prompt": prompt,
                **metrics,
            }

            # Derive training run name and fetch benchmark results from S3
            training_run = derive_training_run_name(run_dir.name, source_dataset)
            record["training_run"] = training_run
            record["results"] = fetch_benchmark_results(training_run, fs) if training_run else None

            results.append(record)

    # Filter out runs missing DCLM scores
    results = [
        r for r in results
        if "input_dclm_score" in r
        and "output_dclm_score" in r
        and "dclm_score_difference" in r
    ]

    output_path = Path(__file__).parent.parent.parent / "rephrasing_metadata.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Wrote {len(results)} run records to {output_path}")


if __name__ == "__main__":
    main()
