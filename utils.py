import os
from datatrove.pipeline.base import PipelineStep


USER = os.environ.get('USER')
PROJECT_NAME = "finephrase"

BASE_PATH = f"/fsx/{USER}"
PROJECT_PATH = f"{BASE_PATH}/projects/{PROJECT_NAME}"
LOG_BASE_PATH = f"{BASE_PATH}/logs/{PROJECT_NAME}/experiments"
S3_BASE_PATH = f"s3://{PROJECT_NAME}/experiments"
LOCAL_TMP_PATH_ON_NODE = f"/scratch/{USER}/tmp/{PROJECT_NAME}"

FAULTY_NODES = "ip-26-0-160-103,ip-26-0-160-242,ip-26-0-161-138,ip-26-0-161-178,ip-26-0-162-46,ip-26-0-162-180"


# -----------------------------
# Reader utilities
# -----------------------------

def get_reader(path):
    from datatrove.pipeline.readers import JsonlReader, ParquetReader

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


# -----------------------------
# Fineweb EDU score utilities
# -----------------------------

_EDU_TOKENIZER = None
_EDU_MODEL = None


def _load_edu_classifier():
    """Lazily load and cache the fineweb-edu classifier."""
    global _EDU_TOKENIZER, _EDU_MODEL
    if _EDU_TOKENIZER is None or _EDU_MODEL is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_repo = "HuggingFaceFW/fineweb-edu-classifier"
        _EDU_TOKENIZER = AutoTokenizer.from_pretrained(model_repo)
        _EDU_MODEL = AutoModelForSequenceClassification.from_pretrained(model_repo)
    return _EDU_TOKENIZER, _EDU_MODEL


def calculate_edu_score(text: str) -> float:
    """Return the fineweb-edu score in [0, 5] for given text."""
    if not text or not text.strip():
        return 0.0
    try:
        tokenizer, model = _load_edu_classifier()
        inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1).float().detach().numpy()
        score = float(logits.item())
        return max(0.0, min(score, 5.0))
    except Exception:
        return 0.0


def calculate_edu_score_dict(text: str) -> dict:
    """Return a dict with continuous and integer fineweb-edu scores."""
    score = calculate_edu_score(text)
    int_score = int(round(max(0.0, min(score, 5.0))))
    return {"score": score, "int_score": int_score}


class EduScoreStatsLogger(PipelineStep):
    """
    Pipeline step that logs education score statistics from document metadata.
    """

    type = "📊 - STATS"
    name = "Education Score Stats Logger"

    def __init__(self):
        super().__init__()

    def run(self, data, rank: int = 0, world_size: int = 1):
        """
        Log education score statistics and pass documents through unchanged.

        Args:
            data: Input documents pipeline
            rank: Worker rank
            world_size: Total number of workers

        Yields:
            Documents unchanged after logging stats
        """
        for doc in data:
            with self.track_time():
                # Extract education scores from metadata
                input_edu_score = doc.metadata.get("input", {}).get("score", 0.0)
                thinking_edu_score = doc.metadata.get("thinking", {}).get("score", 0.0)
                output_edu_score = doc.metadata.get("score", 0.0)
                edu_score_difference = doc.metadata.get("edu_score_difference", 0.0)
                edu_score_improvement = doc.metadata.get("edu_score_improvement", 0)

                input_token_count = doc.metadata.get("input", {}).get("token_count", 0)
                thinking_token_count = doc.metadata.get("thinking", {}).get("token_count", 0)
                output_token_count = doc.metadata.get("token_count", 0)
                token_reduction = doc.metadata.get("token_reduction", 0)

                # Log the statistics
                self.stat_update("input_edu_score", value=input_edu_score)
                self.stat_update("thinking_edu_score", value=thinking_edu_score)
                self.stat_update("output_edu_score", value=output_edu_score)
                self.stat_update("edu_score_difference", value=edu_score_difference)
                self.stat_update("edu_score_improvement", value=edu_score_improvement)

                self.stat_update("input_token_count", value=input_token_count)
                self.stat_update("thinking_token_count", value=thinking_token_count)
                self.stat_update("output_token_count", value=output_token_count)
                self.stat_update("token_reduction", value=token_reduction)

            yield doc