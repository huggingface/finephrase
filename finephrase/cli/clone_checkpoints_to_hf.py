"""Clone FinePhrase training checkpoints from S3 into an HF Bucket.

Discovers run folders under ``s3://finephrase/experiments/checkpoints/`` and clones the
ones not yet present in the destination HF Bucket, via the cluster-native
``clone-s3-to-hf.sh`` cloner (slurm-array sharded, streams S3 -> xet, no disk staging).

By default it clones every run folder EXCEPT those whose name contains one of
``EXCLUDE_MARKERS`` (decay-resumed grids, rephrase-budget ablations, essentialweb raw
datasets), and skips any run already present in the bucket. Pass ``--runs`` to clone an
explicit set instead (ignoring both filters).

Why we clone from the *parent* prefix and exclude everything else:
    The cloner derives each destination key by stripping the **exact** ``--source``
    prefix from the S3 key. The bucket layout is top-level ``<run>/...`` folders, which
    only comes out right when the source is the shared parent
    ``s3://finephrase/experiments/checkpoints/`` (leaving ``<run>/...`` as the key). A
    per-run source would strip the run folder and collapse runs onto each other. So we
    always clone from the parent and pass one ``--exclude`` glob per run we do NOT want.

Why we skip runs already in the bucket:
    The cloner does NOT diff against the bucket -- it re-reads its whole source set every
    run. By default we feed it only the runs whose folder does not yet exist in the bucket
    (one cheap bucket API call per run). A run left half-cloned by an interrupted job
    counts as "present" and is skipped; use ``--repair`` to recover those.

``--repair`` (precise, file-level):
    Lists the actual S3 vs bucket file keys, computes the *missing* files, and submits a
    clone restricted to only the FNV shards (``fnv1a64(key) % 64``) and runs that contain
    missing files. Because shard failures lose whole FNV buckets, this re-reads only those
    buckets (~the missing fraction) instead of re-streaming every run. The S3 *listing*
    (~1.7M keys) is slow, but the actual transfer is minimal.

This clones (copies) the checkpoints; it never deletes the S3 source.

Usage::

    # Clone every non-decay run not yet present in the bucket (default; prints plan):
    python -m finephrase.cli.clone_checkpoints_to_hf

    # Submit it:
    python -m finephrase.cli.clone_checkpoints_to_hf --submit

    # Repair: copy only the files that differ between S3 and the bucket:
    python -m finephrase.cli.clone_checkpoints_to_hf --repair --submit

    # Clone an explicit subset instead (e.g. a single run), regardless of bucket state:
    python -m finephrase.cli.clone_checkpoints_to_hf --runs fw_edu_hq --submit
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path

import requests
from fsspec.core import url_to_fs
from joblib import Parallel, delayed
from tqdm import tqdm

from finephrase.utils import S3_BASE_PATH

logger = logging.getLogger(__name__)

CLONER = "/admin/scripts/cloner/clone-s3-to-hf.sh"
CLONER_BIN = "/admin/scripts/cloner/hf-bucket-cloner"
SBATCH = "/opt/slurm/bin/sbatch"
DEFAULT_DEST = "HuggingFaceFW/finephrase-checkpoints"
DEFAULT_HUB = "https://huggingface.co"
DEFAULT_SHARDS = 64
# Repair re-shards at this fixed count; needed shard ids are derived from the missing keys'
# fnv1a64 % REPAIR_SHARD_COUNT, so it is independent of the original jobs' shard counts.
REPAIR_SHARD_COUNT = 64
N_JOBS = 32

# All trained-model checkpoints live under this single shared S3 prefix.
CHECKPOINTS_S3 = f"{S3_BASE_PATH}/checkpoints"
# Run-name substrings for families we never mirror to the bucket by default: decay-resumed
# grids, rephrase-budget ablations, and the essentialweb raw datasets.
EXCLUDE_MARKERS = ("-decay-", "rephrase", "essentialweb")
HF_TOKEN_PATH = Path.home() / ".cache/huggingface/token"
LOG_DIR = Path(f"/fsx/{os.environ['USER']}/hf-bucket-cloner/logs")


def fnv1a64(text: str) -> int:
    """64-bit FNV-1a hash, matching the cloner's ``fnv1a64(key) % shard_count`` sharding."""
    h = 0xCBF29CE484222325
    for byte in text.encode():
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def list_s3_run_prefixes(checkpoints_s3: str) -> set[str]:
    """List the run-folder names directly under the S3 checkpoints prefix."""
    fs, root = url_to_fs(checkpoints_s3)
    return {entry.rstrip("/").split("/")[-1] for entry in fs.ls(root, detail=False)}


def bucket_run_present(dest: str, hub: str, token: str, run: str) -> bool:
    """Whether ``<run>/`` has at least one file in the bucket (one page is enough)."""
    resp = requests.get(
        f"{hub}/api/buckets/{dest}/tree/{run}?recursive=true",
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    resp.raise_for_status()
    return len(resp.json()) > 0


def bucket_run_keys(dest: str, hub: str, token: str, run: str) -> set[str]:
    """All file paths (``<run>/...``) under a run in the bucket, following pagination."""
    headers = {"Authorization": f"Bearer {token}"}
    url: str | None = f"{hub}/api/buckets/{dest}/tree/{run}?recursive=true"
    keys: set[str] = set()
    while url:
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
        keys.update(entry["path"] for entry in resp.json() if entry["type"] == "file")
        url = resp.links["next"]["url"] if "next" in resp.links else None
    return keys


def s3_run_keys(checkpoints_s3: str) -> dict[str, set[str]]:
    """Map run -> set of well-formed S3 keys (bucket-relative, the cloner's hash input).

    Skips empty-path-segment keys (e.g. ``fw_edu_hq//1000/...``) the cloner drops. A single
    recursive listing of ~1.7M objects is slow (minutes) but only happens in ``--repair``.
    """
    fs, root = url_to_fs(checkpoints_s3)
    s3_bucket = root.split("/", 1)[0]
    runs: dict[str, set[str]] = {}
    prefix_len = len(root) + 1  # strip "<s3_bucket>/experiments/checkpoints/" to get "<run>/..."
    for key in fs.find(root):
        if "//" in key:
            continue
        run = key[prefix_len:].split("/", 1)[0]
        runs.setdefault(run, set()).add(
            key[len(s3_bucket) + 1 :]
        )  # "experiments/checkpoints/<run>/..."
    return runs


def select_missing_runs(candidates: list[str], dest: str, hub: str, token: str) -> list[str]:
    """Return candidate runs whose folder does not yet exist in the bucket (fast)."""

    def check(run: str) -> tuple[str, bool]:
        return run, bucket_run_present(dest, hub, token, run)

    decisions = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(check)(run) for run in tqdm(candidates, desc="Checking bucket")
    )
    return [run for run, present in decisions if not present]


def diff_missing_files(
    candidates: list[str], dest: str, hub: str, token: str, key_prefix: str
) -> tuple[list[str], list[int], int]:
    """File-level S3-vs-bucket diff: which runs/shards hold files missing from the bucket.

    Returns (runs_with_missing, needed_shard_ids, missing_file_count). ``needed_shard_ids``
    are the distinct ``fnv1a64(key) % REPAIR_SHARD_COUNT`` of every missing file.
    """
    logger.info("Enumerating S3 keys (~1.7M objects, can take a few minutes)...")
    s3_keys = s3_run_keys(CHECKPOINTS_S3)

    def diff(run: str) -> set[str]:
        present = bucket_run_keys(dest, hub, token, run)
        run_keys = s3_keys[run] if run in s3_keys else set()
        # A run key is "<key_prefix>/<run>/..."; its bucket path is the part after key_prefix/.
        return {key for key in run_keys if key[len(key_prefix) + 1 :] not in present}

    missing_per_run = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(diff)(run) for run in tqdm(candidates, desc="Diffing files (S3 vs bucket)")
    )

    runs_with_missing = sorted(run for run, missing in zip(candidates, missing_per_run) if missing)
    needed_shards = sorted(
        {fnv1a64(key) % REPAIR_SHARD_COUNT for missing in missing_per_run for key in missing}
    )
    missing_count = sum(len(missing) for missing in missing_per_run)
    return runs_with_missing, needed_shards, missing_count


def build_clone_command(
    *, dest: str, shards: int, exclude_runs: list[str], key_prefix: str, submit: bool
) -> list[str]:
    """Assemble the ``clone-s3-to-hf.sh`` argv: parent source plus one exclude per run.

    Excludes match the full S3 key, so each run is anchored as ``<key_prefix>/<run>/*`` --
    the trailing slash stops a run name matching a longer sibling (``fw_edu_hq/`` vs
    ``fw_edu_hq--...``).
    """
    command = [CLONER, "-s", f"{CHECKPOINTS_S3}/", "-d", dest, "-n", str(shards)]
    for run in exclude_runs:
        command += ["--exclude", f"{key_prefix}/{run}/*"]
    if submit:
        command.append("-y")  # we have already shown and confirmed the plan
    return command


def build_repair_script(*, dest: str, exclude_runs: list[str], key_prefix: str) -> str:
    """Generate a self-contained sbatch worker that clones one fixed FNV shard.

    Mirrors the driver's worker env (AWS creds, node-local HF_HOME, xet concurrency) but
    pins ``--shard-count`` to ``REPAIR_SHARD_COUNT`` and reads ``--shard-id`` from the array
    task, so a sparse ``--array`` re-runs exactly the buckets that hold missing files.
    """
    excludes = "\n".join(f"  --exclude '{key_prefix}/{run}/*'" for run in exclude_runs)
    return f"""#!/bin/bash
set -uo pipefail
export PATH="/opt/slurm/bin:$PATH" NO_COLOR=1
eval "$(aws configure export-credentials --format env)"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export HF_HOME="/scratch/hf-repair-${{SLURM_JOB_ID}}-${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$HF_HOME"; trap 'rm -rf "$HF_HOME"' EXIT
export HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY=32
export HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY=64
export HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION=32
EXCLUDES=(
{excludes}
)
echo "=== repair shard ${{SLURM_ARRAY_TASK_ID}}/{REPAIR_SHARD_COUNT} on $(hostname) ==="
"{CLONER_BIN}" \\
  --parallel-files 32 --s3-part-concurrency 8 --s3-part-size-mib 16 \\
  --shard-id "$SLURM_ARRAY_TASK_ID" --shard-count {REPAIR_SHARD_COUNT} \\
  "${{EXCLUDES[@]}}" \\
  "{CHECKPOINTS_S3}/" "{dest}"
"""


def submit_repair(
    *, dest: str, exclude_runs: list[str], needed_shards: list[int], key_prefix: str, submit: bool
) -> None:
    """Write the repair worker script and submit it as a sparse slurm array."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    script_path = LOG_DIR.parent / "repair-worker.sh"
    script_path.write_text(
        build_repair_script(dest=dest, exclude_runs=exclude_runs, key_prefix=key_prefix)
    )
    array = ",".join(str(shard) for shard in needed_shards)
    command = [
        SBATCH,
        "--job-name=clone-repair",
        "--partition=hopper-cpu",
        f"--array={array}",
        "--cpus-per-task=8",
        "--mem=32G",
        "--time=4:00:00",
        "--requeue",
        f"--output={LOG_DIR}/%x-%A_%a.out",
        f"--error={LOG_DIR}/%x-%A_%a.err",
        str(script_path),
    ]
    if not submit:
        logger.info(
            "Dry run (pass --submit to launch). Repair worker: %s\nsbatch:\n%s",
            script_path,
            shlex.join(command),
        )
        return
    logger.info("Submitting repair array (%d shards) via %s", len(needed_shards), SBATCH)
    subprocess.run(command, check=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dest", type=str, default=DEFAULT_DEST, help="Destination HF bucket (org/name)."
    )
    parser.add_argument(
        "--shards", type=int, default=DEFAULT_SHARDS, help="Slurm array size (default mode only)."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Explicit run folders to clone, overriding discovery (ignores the name-marker "
        "and already-in-bucket filters).",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Precise file-level diff: copy only the files missing from the bucket "
        "(re-runs just the FNV shards that hold them).",
    )
    parser.add_argument(
        "--submit", action="store_true", help="Submit the clone; otherwise only print the plan."
    )
    args = parser.parse_args()

    # S3 object keys are relative to the bucket, e.g. "experiments/checkpoints/<run>/...".
    key_prefix = CHECKPOINTS_S3.split("/", 3)[3]
    s3_runs = list_s3_run_prefixes(CHECKPOINTS_S3)
    token = HF_TOKEN_PATH.read_text().strip()

    if args.repair:
        candidates = sorted(run for run in s3_runs if not any(m in run for m in EXCLUDE_MARKERS))
        runs_with_missing, needed_shards, missing_count = diff_missing_files(
            candidates, args.dest, DEFAULT_HUB, token, key_prefix
        )
        logger.info("Missing files: %d across %d runs", missing_count, len(runs_with_missing))
        logger.info(
            "Needed FNV shards: %d/%d (re-reads ~%.0f%% of those runs): %s",
            len(needed_shards),
            REPAIR_SHARD_COUNT,
            100 * len(needed_shards) / REPAIR_SHARD_COUNT,
            needed_shards,
        )
        if not needed_shards:
            raise SystemExit("Nothing to repair: bucket already matches S3 for all candidate runs.")
        exclude_runs = sorted(s3_runs - set(runs_with_missing))
        submit_repair(
            dest=args.dest,
            exclude_runs=exclude_runs,
            needed_shards=needed_shards,
            key_prefix=key_prefix,
            submit=args.submit,
        )
        return

    if args.runs:
        requested = set(args.runs)
        to_clone = sorted(requested & s3_runs)
        missing = sorted(requested - s3_runs)
        if missing:
            logger.warning(
                "%d requested run(s) absent on S3 (skipped): %s", len(missing), ", ".join(missing)
            )
    else:
        candidates = sorted(run for run in s3_runs if not any(m in run for m in EXCLUDE_MARKERS))
        logger.info(
            "S3 run folders: %d (%d candidates, %d skipped by markers %s)",
            len(s3_runs),
            len(candidates),
            len(s3_runs) - len(candidates),
            EXCLUDE_MARKERS,
        )
        to_clone = select_missing_runs(candidates, args.dest, DEFAULT_HUB, token)
        logger.info("Already present in bucket (skipped): %d", len(candidates) - len(to_clone))

    exclude_runs = sorted(s3_runs - set(to_clone))
    logger.info("Will clone: %d runs -> %s", len(to_clone), args.dest)
    logger.info("Will exclude (non-target S3 prefixes): %d", len(exclude_runs))
    if not to_clone:
        raise SystemExit(
            "Nothing to clone: selected runs are already in the bucket (or none matched)."
        )

    command = build_clone_command(
        dest=args.dest,
        shards=args.shards,
        exclude_runs=exclude_runs,
        key_prefix=key_prefix,
        submit=args.submit,
    )
    if not args.submit:
        logger.info("Dry run (pass --submit to launch). Cloner command:\n%s", shlex.join(command))
        return
    logger.info("Submitting clone via %s", CLONER)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
