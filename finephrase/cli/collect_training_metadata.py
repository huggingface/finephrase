"""Collect metadata for proportion-sweep and variance-check training runs.

For each run dir under ``LOG_BASE_PATH/training/`` that matches one of the
``EXPERIMENT_FAMILIES`` patterns, this reads:
  - the local training ``config.yaml`` (datasets, weights, seeds, train_steps)
  - the final-step (``FINAL_CHECKPOINT``) lighteval JSON from S3

and writes a single ``training_metadata.json`` at the project root.

Schema for each record:
  - run: str                  training run name (== checkpoint folder on S3)
  - experiment_family: str    e.g. "proportion_sweep" or "variance_check"
  - model_size: str           "0.5b" / "1.7b" / "2.9b" / "6.2b"
  - datasets: list[str]       short names (last path segment of dataset_folder)
  - data_weights: list[float] parallel to ``datasets``
  - seed: int                 model init / training RNG seed
  - data_seed: int            data ordering seed
  - train_steps: int
  - tokens_consumed_b: float  total training tokens in billions
  - results: dict | None      same shape as ``rephrasing_metadata.json``
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import yaml
from fsspec.core import url_to_fs
from tqdm import tqdm

from finephrase.benchmark import S3_EVALS_PATH, fetch_benchmark_results
from finephrase.utils import LOG_BASE_PATH

logger = logging.getLogger(__name__)

BASE_PATH = Path(LOG_BASE_PATH) / "training"

# (compiled regex, family). First match wins. Extend here when a new family appears.
EXPERIMENT_FAMILIES: list[tuple[re.Pattern[str], str]] = [
    # mix-0.X-fw_edu_hq-0.Y-<format>_smollm2_1.7b_hq (X+Y = 1.0, formats: math/table/faq/tutorial).
    (re.compile(r"^mix-0\.[1-9]-fw_edu_hq-0\.[1-9]-\w+_smollm2_1\.7b_hq$"), "proportion_sweep"),
    # Any 3x3 seed x data-seed grid cell (launcher appends --data-seed=N-seed=M, train_model appends -seed-NM0M).
    (re.compile(r"--data-seed=\d+-seed=\d+-seed-\d+$"), "variance_check"),
]

# Model-size suffix as a complete token. Must be preceded by `-` and followed by `-` or end of name.
_MODEL_SIZE_RE = re.compile(r"-(0\.5|1\.7|2\.9|6\.2)b(?=-|$)")
DEFAULT_MODEL_SIZE = "1.7b"


def classify(run_name: str) -> str | None:
    """Return the matching experiment family, or None to skip the run."""
    for pattern, family in EXPERIMENT_FAMILIES:
        if pattern.search(run_name):
            return family
    return None


def infer_model_size(run_name: str) -> str:
    """Map a run name to one of the QWEN_SIZE_PRESETS keys (defaults to 1.7b)."""
    m = _MODEL_SIZE_RE.search(run_name)
    return f"{m.group(1)}b" if m else DEFAULT_MODEL_SIZE


def _short_dataset_name(path: str) -> str:
    """Take the last path segment (datasets live under .../dataset/<name>/)."""
    return path.rstrip("/").split("/")[-1]


def parse_config(config_path: Path) -> dict:
    """Pull the training-relevant fields out of a nanotron config.yaml.

    The first ``data_stages`` entry is taken as authoritative (decay runs duplicate
    the same dataset config across stages; non-decay runs only have one stage).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    stage = cfg["data_stages"][0]["data"]
    ds_cfg = stage["dataset"]
    folders: list[str] = list(ds_cfg["dataset_folder"])

    # Older configs may omit dataset_weights → uniform per the project's convention.
    raw_weights = ds_cfg["dataset_weights"] if "dataset_weights" in ds_cfg else [1.0 / len(folders)] * len(folders)
    weights = [float(w) for w in raw_weights]
    if len(weights) != len(folders):
        raise ValueError(
            f"dataset_folder ({len(folders)}) and dataset_weights ({len(weights)}) mismatch in {config_path}"
        )

    micro = cfg["tokens"]["micro_batch_size"]
    accum = cfg["tokens"]["batch_accumulation_per_replica"]
    dp = cfg["parallelism"]["dp"]
    seq = cfg["tokens"]["sequence_length"]
    train_steps = int(cfg["tokens"]["train_steps"])
    global_batch = micro * accum * dp
    tokens_consumed = train_steps * global_batch * seq

    return {
        "datasets": [_short_dataset_name(p) for p in folders],
        "data_weights": weights,
        "seed": int(cfg["general"]["seed"]),
        "data_seed": int(stage["seed"]),
        "train_steps": train_steps,
        "tokens_consumed_b": round(tokens_consumed / 1e9, 3),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fs, _ = url_to_fs(S3_EVALS_PATH)
    if not BASE_PATH.exists():
        raise SystemExit(f"Training logs directory not found: {BASE_PATH}")

    run_dirs = sorted(d for d in BASE_PATH.iterdir() if d.is_dir())
    records: list[dict] = []
    for run_dir in tqdm(run_dirs, desc="Scanning training runs"):
        family = classify(run_dir.name)
        if family is None:
            continue
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            logger.warning("Skipping %s: no config.yaml", run_dir.name)
            continue
        try:
            parsed = parse_config(config_path)
        except (KeyError, ValueError, yaml.YAMLError) as exc:
            logger.warning("Skipping %s: failed to parse config.yaml (%s)", run_dir.name, exc)
            continue
        results = fetch_benchmark_results(run_dir.name, fs)
        records.append(
            {
                "run": run_dir.name,
                "experiment_family": family,
                "model_size": infer_model_size(run_dir.name),
                **parsed,
                "results": results,
            }
        )

    records.sort(key=lambda r: (r["experiment_family"], r["run"]))
    out_path = Path(__file__).parent.parent.parent / "training_metadata.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    by_family: dict[str, int] = {}
    by_family_with_results: dict[str, int] = {}
    for r in records:
        by_family[r["experiment_family"]] = by_family.get(r["experiment_family"], 0) + 1
        if r["results"] is not None:
            by_family_with_results[r["experiment_family"]] = by_family_with_results.get(r["experiment_family"], 0) + 1
    logger.info("Wrote %d records to %s", len(records), out_path)
    for family in sorted(by_family):
        logger.info("  %s: %d records (%d with eval results)", family, by_family[family], by_family_with_results.get(family, 0))


if __name__ == "__main__":
    main()
