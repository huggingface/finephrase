"""Collect metadata from rephrasing runs into a structured JSON file.

Walks beyondweb/, format/, nemotron/, rewire/ folders under the rephrasing logs dir,
filters to runs with >= 90 completions, loads or aggregates stats, and extracts
token counts, quality scores, model, source dataset, and prompt info.
"""

import json
import logging
import re
from pathlib import Path

from tqdm import tqdm

from datatrove.utils.stats import PipelineStats

logger = logging.getLogger(__name__)

BASE_PATH = Path("/fsx/joel_niklaus/logs/finephrase/experiments/rephrasing")
CATEGORIES = ["beyondweb", "format", "nemotron", "rewire"]
MIN_COMPLETIONS = 90

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


def count_completions(run_dir: Path) -> int:
    """Count entries in the completions directory."""
    completions_dir = run_dir / "completions"
    if not completions_dir.exists():
        return 0
    return len(list(completions_dir.iterdir()))


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


def _stat_value(stats: dict, key: str, field: str) -> float | None:
    """Extract a field from a stat entry that may be a dict or scalar."""
    value = stats.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        return value[field]
    return value


def _find_stage(stats: list[dict], name_substring: str) -> dict | None:
    """Find the first pipeline stage whose name contains the given substring."""
    for stage in stats:
        if name_substring in stage["name"]:
            return stage
    return None


def extract_metrics(stats: list[dict]) -> dict[str, float | None]:
    """Extract token counts and quality scores from aggregated pipeline stats."""
    metrics: dict[str, float | None] = {}

    # Token counts and runtime from the inference stage
    inference = _find_stage(stats, "Inference")
    if inference:
        s = inference["stats"]
        metrics["prompt_tokens"] = _stat_value(s, "prompt_tokens", "total")
        metrics["completion_tokens"] = _stat_value(s, "completion_tokens", "total")
        metrics["failed_requests"] = _stat_value(s, "failed_requests", "total")

        # Total runtime in hours and output throughput
        runtime_seconds = inference["time_stats"]["total"]
        metrics["total_runtime_hours"] = round(runtime_seconds / 3600, 2)
        completion_tokens = metrics["completion_tokens"]
        if completion_tokens and runtime_seconds > 0:
            metrics["output_tokens_per_second"] = round(completion_tokens / runtime_seconds, 1)

        # Compression ratio: output / input tokens
        prompt_tokens = metrics["prompt_tokens"]
        if prompt_tokens and completion_tokens:
            metrics["compression_ratio"] = round(completion_tokens / prompt_tokens, 4)

    # Number of input documents from the first reader stage
    reader = _find_stage(stats, "READER")
    if reader:
        metrics["num_documents"] = _stat_value(reader["stats"], "documents", "total")

    # Quality scores from the quality/education stats logger stage
    quality = _find_stage(stats, "Quality Score") or _find_stage(stats, "Education Score")
    if quality:
        s = quality["stats"]
        metrics["input_edu_score"] = _stat_value(s, "input_edu_score", "mean")
        metrics["output_edu_score"] = _stat_value(s, "output_edu_score", "mean")
        metrics["edu_score_difference"] = _stat_value(s, "edu_score_difference", "mean")
        metrics["edu_score_improvement"] = _stat_value(s, "edu_score_improvement", "mean")
        metrics["input_dclm_score"] = _stat_value(s, "input_dclm_score", "mean")
        metrics["output_dclm_score"] = _stat_value(s, "output_dclm_score", "mean")
        metrics["dclm_score_difference"] = _stat_value(s, "dclm_score_difference", "mean")
        metrics["dclm_score_improvement"] = _stat_value(s, "dclm_score_improvement", "mean")

    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results: list[dict] = []

    for category in CATEGORIES:
        category_dir = BASE_PATH / category
        if not category_dir.exists():
            logger.warning(f"Category directory not found: {category_dir}")
            continue

        run_dirs = sorted(d for d in category_dir.iterdir() if d.is_dir())

        for run_dir in tqdm(run_dirs, desc=f"Processing {category}"):
            num_completions = count_completions(run_dir)
            if num_completions < MIN_COMPLETIONS:
                logger.info(
                    f"Skipping {category}/{run_dir.name}: only {num_completions} completions"
                )
                continue

            stats = load_or_aggregate_stats(run_dir)
            if stats is None:
                logger.warning(f"No stats found for {category}/{run_dir.name}")
                continue

            metrics = extract_metrics(stats)

            try:
                model, source_dataset = extract_model_and_dataset(run_dir)
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Could not read executor.json for {category}/{run_dir.name}: {e}"
                )
                continue

            prompt = derive_prompt(category, run_dir.name)

            results.append({
                "run": f"{category}/{run_dir.name}",
                "model": model,
                "source_dataset": source_dataset,
                "prompt": prompt,
                "num_completions": num_completions,
                **metrics,
            })

    # Filter out runs missing DCLM scores
    results = [
        r for r in results
        if r.get("input_dclm_score") is not None
        and r.get("output_dclm_score") is not None
        and r.get("dclm_score_difference") is not None
    ]

    output_path = Path(__file__).parent / "rephrasing_metadata.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Wrote {len(results)} run records to {output_path}")


if __name__ == "__main__":
    main()
