import os
from datatrove.pipeline.readers import JsonlReader, ParquetReader


USER = os.environ.get('USER')
PROJECT_NAME = "finephrase"

BASE_PATH = f"/fsx/{USER}"
PROJECT_PATH = f"{BASE_PATH}/projects/{PROJECT_NAME}"
LOG_BASE_PATH = f"{BASE_PATH}/logs/{PROJECT_NAME}/experiments"
S3_BASE_PATH = f"s3://{PROJECT_NAME}/experiments"
LOCAL_TMP_PATH_ON_NODE = f"/scratch/{USER}/tmp/{PROJECT_NAME}"

def get_reader(path):
    if path.startswith("hf://"):
        # hf datasets are usually in Parquet format
        return ParquetReader
    elif path.startswith("s3://"):
        # s3 datasets are usually in JSONL format
        return JsonlReader
    else:
        raise ValueError(f"Invalid path: {path}")


def build_reader(path: str, *, limit: int, n_tasks: int, shuffle_files: bool = True, text_key: str = "text"):
    """Construct a reader instance with a per-task limit.

    Args:
        path: Dataset path (hf://... or s3://...)
        limit: Global limit across all tasks (-1 means unlimited)
        n_tasks: Total number of tasks across which to shard the limit
        shuffle_files: Whether to shuffle input files
        text_key: Text field key for JSONL inputs (ignored by parquet)

    Returns:
        An initialized reader instance
    """
    per_task_limit = -1 if limit < 0 else limit // n_tasks
    return get_reader(path)(path, shuffle_files=shuffle_files, limit=per_task_limit, text_key=text_key)


def human_readable(num):
    if num >= 1_000_000_000_000:
        return f"~{round(num / 1_000_000_000_000)}T"
    elif num >= 1_000_000_000:
        return f"~{round(num / 1_000_000_000)}B"
    elif num >= 1_000_000:
        return f"~{round(num / 1_000_000)}M"
    elif num >= 1_000:
        return f"~{round(num / 1_000)}K"
    else:
        return f"{num}"