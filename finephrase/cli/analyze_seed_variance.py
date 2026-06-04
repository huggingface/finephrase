"""Analyse run-to-run variance of the seed x data-seed grids.

For each config this loads the step-10000 ``agg_score_micro`` for the 3x3
``data-seed`` x ``seed`` grid, reports the spread, and splits the variance into a
seed share (model init + training RNG) and a data-seed share (data ordering) via
a balanced two-way decomposition. It also checks whether each mix's gain over its
baseline exceeds the run-to-run noise.

Two grids are analysed and reported separately: the full-training grid (trained
from scratch) and the cheaper decay grid (resumed from a shared step-9000
checkpoint). Keeping them apart matters because the shared decay checkpoint fixes
the model init, so the decay grid cannot expose seed/init variance.

Run with: ``python -m finephrase.cli.analyze_seed_variance``
"""

import numpy as np
from fsspec.core import url_to_fs
from loguru import logger

from finephrase.benchmark import S3_EVALS_PATH, fetch_benchmark_results

METRIC = "agg_score_micro"
SEEDS = (1, 2, 3)
DATA_SEEDS = (1, 2, 3)

# Full-training scale ablation: table_mix vs fw_edu_hq baseline at each model
# size. Each entry is (size_label, {config_label: base_run}, baseline_label).
# The 1.7b runs predate the size-suffix convention, hence the bare base names.
FULL_SCALE_GROUPS: list[tuple[str, dict[str, str], str]] = [
    ("0.5b", {
        "table_mix (0.5b)": "mix-fw_edu_hq-table_smollm2_1.7b_hq-0.5b",
        "baseline (0.5b)": "fw_edu_hq-0.5b",
    }, "baseline (0.5b)"),
    ("1.7b", {
        "table_mix (1.7b)": "mix-fw_edu_hq-table_smollm2_1.7b_hq",
        "baseline (1.7b)": "fw_edu_hq",
    }, "baseline (1.7b)"),
    ("2.9b", {
        "table_mix (2.9b)": "mix-fw_edu_hq-table_smollm2_1.7b_hq-2.9b",
        "baseline (2.9b)": "fw_edu_hq-2.9b",
    }, "baseline (2.9b)"),
    ("6.2b", {
        "table_mix (6.2b)": "mix-fw_edu_hq-table_smollm2_1.7b_hq-6.2b",
        "baseline (6.2b)": "fw_edu_hq-6.2b",
    }, "baseline (6.2b)"),
]

# Decay grids (resumed from s3://.../checkpoints/fw_edu_lq/9000/). Shared init.
DECAY_CONFIGS: dict[str, str] = {
    "decay baseline (fw_edu_hq)": "fw_edu_lq-decay-fw_edu_hq",
    "decay mix (+tutorial_12b_hq)": "fw_edu_lq-decay-mix-fw_edu_hq-tutorial_12b_hq",
    "decay mix (+continue_1b_hq)": "fw_edu_lq-decay-mix-fw_edu_hq-continue_1b_hq",
}
DECAY_BASELINE = "decay baseline (fw_edu_hq)"


def grid_run_name(base: str, data_seed: int, seed: int) -> str:
    """Reconstruct the launched run name for a (data_seed, seed) grid cell."""
    return f"{base}--data-seed={data_seed}-seed={seed}-seed-{data_seed * 100 + seed}"


def fetch_micro_optional(run: str, fs: object) -> float | None:
    """Return the step-10000 ``agg_score_micro`` for a run, or None if not evaluated."""
    scores = fetch_benchmark_results(run, fs)
    return scores[METRIC] if scores is not None and METRIC in scores else None


def load_grid(base: str, fs: object) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Load the 3x3 score matrix (rows = data-seed, cols = seed).

    Cells without a step-10000 eval are filled with NaN and returned in a
    ``missing`` list so callers can flag incomplete grids explicitly.
    """
    grid = np.full((len(DATA_SEEDS), len(SEEDS)), np.nan)
    missing: list[tuple[int, int]] = []
    for i, d in enumerate(DATA_SEEDS):
        for j, s in enumerate(SEEDS):
            value = fetch_micro_optional(grid_run_name(base, d, s), fs)
            if value is None:
                missing.append((d, s))
            else:
                grid[i, j] = value
    return grid, missing


def variance_split(grid: np.ndarray) -> dict[str, float]:
    """Partition total variance into seed, data-seed, and residual shares.

    Balanced two-way layout without replication (rows = data-seed, cols = seed),
    so the residual term also absorbs any interaction. Shares are descriptive
    sums-of-squares fractions (eta-squared) and always sum to 1.
    """
    n_d, n_s = grid.shape
    grand = grid.mean()
    row_means = grid.mean(axis=1)  # per data-seed
    col_means = grid.mean(axis=0)  # per seed

    ss_total = ((grid - grand) ** 2).sum()
    ss_dataseed = n_s * ((row_means - grand) ** 2).sum()
    ss_seed = n_d * ((col_means - grand) ** 2).sum()
    ss_resid = ss_total - ss_dataseed - ss_seed
    return {
        "frac_seed": float(ss_seed / ss_total),
        "frac_dataseed": float(ss_dataseed / ss_total),
        "frac_residual": float(ss_resid / ss_total),
    }


def _fmt(value: float) -> str:
    """Format a grid cell, showing NaN (not-yet-evaluated) cells as dashes."""
    return "  --  " if np.isnan(value) else f"{value:.4f}"


def report_config(label: str, base: str, fs: object) -> dict[str, float]:
    """Print the per-config grid + spread and return summary stats."""
    grid, missing = load_grid(base, fs)
    default = fetch_micro_optional(base, fs)  # the seed=6/data-seed=6 leaderboard run, if evaluated
    flat = grid[~np.isnan(grid)]

    logger.info(f"\n=== {label} ===")
    logger.info(f"base run: {base}")
    if missing:
        logger.warning(f"incomplete grid: {len(missing)} cell(s) without eval (data-seed,seed)={missing}")
    header = "data-seed \\ seed |" + "".join(f"  seed={s}" for s in SEEDS) + "  | row mean"
    logger.info(header)
    logger.info("-" * len(header))
    for i, d in enumerate(DATA_SEEDS):
        logger.info(
            f"   data-seed={d}    |" + "".join(f"  {_fmt(v)}" for v in grid[i]) + f"  |  {np.nanmean(grid[i]):.4f}"
        )
    logger.info(
        "   col mean      |" + "".join(f"  {np.nanmean(grid[:, j]):.4f}" for j in range(grid.shape[1]))
        + f"  |  {np.nanmean(grid):.4f}"
    )
    std = flat.std(ddof=1)
    logger.info(
        f"n={flat.size}  mean={flat.mean():.4f}  std={std:.4f}  CV={100 * std / flat.mean():.2f}%  "
        f"min={flat.min():.4f}  max={flat.max():.4f}  range={flat.max() - flat.min():.4f}"
    )
    logger.info(
        f"default (seed=6,data-seed=6) anchor: {default:.4f}" if default is not None
        else "default (seed=6,data-seed=6) anchor: not evaluated"
    )
    if missing:
        logger.info("variance share  ->  skipped (grid incomplete)")
    else:
        split = variance_split(grid)
        logger.info(
            f"variance share  ->  seed={100 * split['frac_seed']:.0f}%   "
            f"data-seed={100 * split['frac_dataseed']:.0f}%   "
            f"residual/interaction={100 * split['frac_residual']:.0f}%"
        )
    return {"mean": float(flat.mean()), "std": float(std), "n": int(flat.size), "default": default}


def compare(label: str, a: dict[str, float], baseline_label: str, b: dict[str, float]) -> None:
    """Print the gain of config ``a`` over baseline ``b`` relative to the noise."""
    gap = a["mean"] - b["mean"]
    pooled_sd = np.sqrt((a["std"] ** 2 + b["std"] ** 2) / 2)
    se_gap = np.sqrt(a["std"] ** 2 / a["n"] + b["std"] ** 2 / b["n"])
    logger.info(
        f"{label} - {baseline_label}: gap={gap:+.4f}  pooled run std={pooled_sd:.4f}  "
        f"Cohen's d={gap / pooled_sd:.1f}  gap/SE={gap / se_gap:.1f}"
    )


def analyze_group(title: str, configs: dict[str, str], baseline_label: str, fs: object) -> None:
    """Report every config in a grid group, then each mix's gain vs the baseline."""
    logger.info(f"\n########## {title} ##########")
    summary = {label: report_config(label, base, fs) for label, base in configs.items()}
    logger.info(f"\n=== {title}: gain vs baseline ===")
    for label, stats in summary.items():
        if label != baseline_label:
            compare(label, stats, baseline_label, summary[baseline_label])


def main() -> None:
    fs, _ = url_to_fs(S3_EVALS_PATH)
    for size, configs, baseline_label in FULL_SCALE_GROUPS:
        analyze_group(f"FULL TRAINING {size}", configs, baseline_label, fs)
    analyze_group("DECAY", DECAY_CONFIGS, DECAY_BASELINE, fs)


if __name__ == "__main__":
    main()
