"""N-gram overlap contamination audit between training data and eval benchmarks.

Workflow
--------
1. Build a decontamination n-gram index from the lighteval eval suite
   (``finephrase.task_list``) using
   :class:`datatrove.pipeline.decont.NGramsDecontIndexer`.
2. For each dataset in ``--data``, scan its documents with
   :class:`finephrase.contamination.NGramContaminationAuditor` and emit per-task
   contamination stats to disk. To keep the comparison across datasets fair we
   subsample with a per-dataset ``--sample-rate`` (typically chosen so each
   dataset contributes the same number of tokens, e.g. 5B).
3. Re-running with ``--report-only`` reads every ``stats.json`` produced in step
   2 and writes the canonical JSON report to the project root.

Layout under ``{LOG_BASE_PATH}/contamination``::

    contamination/
      index_n<N>/                      # benchmark n-gram hash files (one per task)
      audits/<audit_name>/
        <dataset_name>/
          logs/...                     # datatrove logging dir
          stats.json                   # merged per-task contamination stats
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.decont import NGramsDecontConfig, NGramsDecontIndexer
from datatrove.pipeline.filters import SamplerFilter

from finephrase.contamination import NGramContaminationAuditor, _benchmark_group
from finephrase.utils import ENV_COMMAND, LOG_BASE_PATH, MAIL_USER, build_reader


# `finephrase.task_list` defines all benchmarks we evaluate on; reused as the
# custom-tasks module passed to lighteval's Registry.
CUSTOM_TASKS_MODULE = "finephrase.task_list"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_FILE = PROJECT_ROOT / "finephrase" / "tasks.txt"
ROOT_JSON_REPORT = PROJECT_ROOT / "contamination_audit_report.json"


# -----------------------------
# Argument parsing helpers
# -----------------------------


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_sample_rates(value: str | None, n: int) -> list[float]:
    """Return a list of sample rates of length ``n``. Single value broadcasts."""
    if not value:
        return [1.0] * n
    rates = [float(v) for v in _split_csv(value)]
    if len(rates) == 1:
        return rates * n
    if len(rates) != n:
        raise ValueError(
            f"--sample-rates has {len(rates)} entries but --data has {n} paths"
        )
    for r in rates:
        if not (0.0 < r <= 1.0):
            raise ValueError(f"sample-rate must be in (0, 1], got {r}")
    return rates


# -----------------------------
# Task name resolution
# -----------------------------


def expand_eval_task_names(tasks_file: Path) -> list[str]:
    """Expand a ``tasks.txt`` (``suite|task|few_shot|truncate`` per line) into
    fully-resolved ``suite|task`` names that the decontamination indexer accepts.

    Supersets like ``lighteval|mmlu_redux_cf`` get expanded into one entry per
    subset (``lighteval|mmlu_redux_cf:abstract_algebra`` etc.).
    """
    from lighteval.tasks.registry import Registry, taskinfo_selector

    registry = Registry(cache_dir=os.getenv("HF_HOME"), custom_tasks=CUSTOM_TASKS_MODULE)
    expanded, _ = taskinfo_selector(str(tasks_file), registry)
    return expanded


# -----------------------------
# Pipeline construction
# -----------------------------


def make_decont_config(args: argparse.Namespace) -> NGramsDecontConfig:
    return NGramsDecontConfig(
        n_grams=args.n_grams,
        find_query_ngrams=args.find_query_ngrams,
        find_overlap_ngrams=args.find_overlap_ngrams,
    )


def build_index(args: argparse.Namespace, index_dir: Path) -> None:
    """Build (once) the n-gram hash index for all configured benchmarks."""
    index_dir.mkdir(parents=True, exist_ok=True)
    tasks = expand_eval_task_names(Path(args.tasks_file))
    print(f"[index] Building n-gram index for {len(tasks)} eval tasks at {index_dir}")
    indexer = NGramsDecontIndexer(
        output_folder=str(index_dir),
        lighteval_tasks=tasks,
        custom_lighteval_tasks=CUSTOM_TASKS_MODULE,
        config=make_decont_config(args),
    )
    LocalPipelineExecutor(pipeline=[indexer], tasks=1, workers=1).run()


def submit_audit(
    args: argparse.Namespace,
    data_path: str,
    name: str,
    sample_rate: float,
    index_dir: Path,
    audit_dir: Path,
) -> None:
    """Submit / run the auditor pipeline for a single dataset."""
    dataset_dir = audit_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    config = make_decont_config(args)

    reader = build_reader(
        data_path,
        limit=args.limit,
        n_tasks=args.n_tasks,
        shuffle_files=True,
        text_key=args.text_key,
        file_type=args.file_type,
        compression=args.compression,
    )
    pipeline: list = [reader]
    if sample_rate < 1.0:
        # SamplerFilter is rate-based; with millions of docs this yields token
        # counts within <1% of the target by the law of large numbers.
        pipeline.append(SamplerFilter(rate=sample_rate, seed=args.sample_seed))
    pipeline.append(NGramContaminationAuditor(index_folder=str(index_dir), config=config))

    job_name = f"audit-{args.audit_name}-{name}"
    print(f"[audit] dataset={name} path={data_path} sample_rate={sample_rate}")
    if args.run_local:
        LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=args.n_tasks,
            workers=args.local_workers,
            logging_dir=str(dataset_dir),
        ).run()
    else:
        SlurmPipelineExecutor(
            pipeline=pipeline,
            tasks=args.n_tasks,
            logging_dir=str(dataset_dir),
            job_name=job_name,
            partition="hopper-cpu",
            qos=args.qos,
            time=args.time,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu_gb=args.mem_per_cpu_gb,
            env_command=ENV_COMMAND,
            mail_user=MAIL_USER,
        ).run()


# -----------------------------
# Reporting
# -----------------------------


def _read_stats(stats_path: Path) -> dict[str, int]:
    """Read a datatrove ``stats.json`` and return ``{stat_name: total}``.

    datatrove's ``stats.json`` is a list of per-step Stats objects. The
    NGramContaminationAuditor is the last step, so its counters live in the
    final entry under ``stats``. Each stat value is either a plain int (when
    only updated with default unit) or a dict with a ``total`` field.
    """
    raw = json.loads(stats_path.read_text())
    auditor_stats = raw[-1]["stats"]
    out: dict[str, int] = {}
    for name, value in auditor_stats.items():
        if isinstance(value, dict):
            out[name] = int(value["total"])
        else:
            out[name] = int(value)
    return out


def _list_index_tasks(index_dir: Path) -> list[str]:
    """List the lighteval task names backed by an existing index."""
    if not index_dir.exists():
        return []
    return sorted(
        p.name.removesuffix(".index.hashes")
        for p in index_dir.glob("*.index.hashes")
    )


def aggregate_report(audit_dir: Path, names: list[str], index_dir: Path | None = None) -> dict:
    """Aggregate per-dataset stats into a single report dict.

    If ``index_dir`` is provided, every task in the index is included in the
    output groups/tasks (even if no training doc matched it), so 0%-contamination
    benchmarks still show up as columns in the report.
    """
    per_dataset: dict[str, dict] = {}
    all_groups: set[str] = set()
    all_tasks: set[str] = set()

    if index_dir is not None:
        for t in _list_index_tasks(index_dir):
            all_tasks.add(t)
            all_groups.add(_benchmark_group(t))

    for name in names:
        stats_path = audit_dir / name / "stats.json"
        if not stats_path.exists():
            print(f"[report] missing stats for {name} at {stats_path}; skipping")
            continue
        stats = _read_stats(stats_path)
        total = stats["total_docs"]
        contaminated = stats["contaminated"] if "contaminated" in stats else 0
        by_group = {
            k.removeprefix("contaminated_group__"): v
            for k, v in stats.items()
            if k.startswith("contaminated_group__")
        }
        by_task = {
            k.removeprefix("contaminated__"): v
            for k, v in stats.items()
            if k.startswith("contaminated__")
        }
        per_dataset[name] = {
            "total_docs": total,
            "contaminated_docs": contaminated,
            "contamination_rate": contaminated / total if total else 0.0,
            "by_group": by_group,
            "by_task": by_task,
        }
        all_groups.update(by_group.keys())
        all_tasks.update(by_task.keys())

    return {
        "datasets": per_dataset,
        "groups": sorted(all_groups),
        "tasks": sorted(all_tasks),
    }


def write_report(report: dict) -> None:
    report_text = json.dumps(report, indent=2)
    ROOT_JSON_REPORT.write_text(report_text)
    print(f"[report] wrote {ROOT_JSON_REPORT}")


# -----------------------------
# Entry point
# -----------------------------


parser = argparse.ArgumentParser(
    "Audit n-gram overlap between training data and eval benchmarks.",
)
parser.add_argument("--data", help="Comma-separated dataset paths to audit")
parser.add_argument("--names", help="Comma-separated names matching --data")
parser.add_argument(
    "--sample-rates",
    default=None,
    help="Comma-separated sample rates in (0, 1] per dataset (default 1.0)",
)
parser.add_argument(
    "--audit-name", required=True, help="Folder name under contamination/audits/"
)
parser.add_argument(
    "--tasks-file",
    default=str(DEFAULT_TASKS_FILE),
    help="Path to lighteval tasks list (suite|task|few_shot|truncate per line)",
)
parser.add_argument("--n-grams", type=int, default=10)
parser.add_argument(
    "--find-query-ngrams",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Include ngrams from the eval input/query in the index. Off by default"
    " because lighteval queries embed prompt-template boilerplate (e.g."
    " 'Question: ... Answer:') that produces a flood of spurious matches.",
)
parser.add_argument(
    "--find-overlap-ngrams",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Include ngrams spanning query and label in the index",
)
parser.add_argument("--n-tasks", type=int, default=200, help="Slurm tasks per dataset")
parser.add_argument(
    "--limit",
    type=int,
    default=-1,
    help="Global document limit per dataset (split across tasks). -1 = no limit.",
)
parser.add_argument("--run-local", action="store_true")
parser.add_argument("--local-workers", type=int, default=8)
parser.add_argument("--qos", default="normal")
parser.add_argument("--time", default="12:00:00")
parser.add_argument("--cpus-per-task", type=int, default=4)
parser.add_argument("--mem-per-cpu-gb", type=int, default=4)
parser.add_argument("--sample-seed", type=int, default=42)
parser.add_argument("--text-key", default="text")
parser.add_argument("--file-type", default=None, choices=[None, "jsonl", "parquet"])
parser.add_argument("--compression", default=None)
parser.add_argument(
    "--skip-index",
    action="store_true",
    help="Reuse an existing index (skip the build step)",
)
parser.add_argument(
    "--report-only",
    action="store_true",
    help="Only aggregate existing stats into contamination_audit_report.json; no jobs submitted",
)


def main() -> None:
    args = parser.parse_args()

    base_dir = Path(f"{LOG_BASE_PATH}/contamination")
    index_dir = base_dir / f"index_n{args.n_grams}"
    audit_dir = base_dir / "audits" / args.audit_name
    audit_dir.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        if not args.data or not args.names:
            parser.error("--data and --names are required (unless --report-only)")
        data_paths = _split_csv(args.data)
        names = _split_csv(args.names)
        if len(data_paths) != len(names):
            parser.error(f"--data has {len(data_paths)} entries, --names has {len(names)}")
        sample_rates = _parse_sample_rates(args.sample_rates, len(data_paths))

        if not args.skip_index:
            build_index(args, index_dir)
        elif not any(index_dir.glob("*.index.hashes")):
            parser.error(f"--skip-index set but no hashes found in {index_dir}")

        for path, name, rate in zip(data_paths, names, sample_rates):
            submit_audit(args, path, name, rate, index_dir, audit_dir)

    # Aggregate whatever stats are already present (works both for --report-only
    # and for --run-local where everything ran synchronously).
    names_for_report = (
        _split_csv(args.names)
        if args.names
        else sorted(p.name for p in audit_dir.iterdir() if p.is_dir())
    )
    report = aggregate_report(audit_dir, names_for_report, index_dir=index_dir)
    if report["datasets"]:
        write_report(report)
    else:
        print(
            "[report] no stats.json files found yet — re-run with --report-only "
            "once slurm jobs have completed."
        )


if __name__ == "__main__":
    main()
