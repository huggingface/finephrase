import os
import logging
import threading
import urllib.request
from pathlib import Path
from datatrove.pipeline.base import PipelineStep

from dotenv import load_dotenv

load_dotenv()

USER = os.environ.get('USER')
PROJECT_NAME = "finephrase"

BASE_PATH = f"/fsx/{USER}"
CACHE_PATH = f"/fsx/{USER}/.cache"  # This is also the HUGGINGFACE_HUB_CACHE
PROJECT_PATH = f"{BASE_PATH}/projects/{PROJECT_NAME}"
CHECKPOINTS_PATH = f"{BASE_PATH}/checkpoints/{PROJECT_NAME}"
LOG_BASE_PATH = f"{BASE_PATH}/logs/{PROJECT_NAME}/experiments"
S3_BASE_PATH = f"s3://{PROJECT_NAME}/experiments"
LOCAL_TMP_PATH_ON_NODE = f"/scratch/{USER}/tmp/{PROJECT_NAME}"

FAULTY_NODES = [
    "ip-26-0-160-103",
    "ip-26-0-160-192",
    "ip-26-0-160-225",
    "ip-26-0-160-242",
    "ip-26-0-161-78",
    "ip-26-0-161-103",
    "ip-26-0-161-138",
    "ip-26-0-161-178",
    "ip-26-0-162-46",
    "ip-26-0-162-180",
    "ip-26-0-164-0",
    "ip-26-0-164-25",
    "ip-26-0-164-236",
    "ip-26-0-169-247",
]
FAULTY_NODES = ",".join(FAULTY_NODES)

# Make sure we're authenticated before running any jobs
ENV_COMMAND = f"sleep $((RANDOM % 30)) && module load cuda/12.4 && hf auth login --token {os.getenv('HF_TOKEN')} && hf auth whoami"

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

def calculate_edu_score(text: str, tokenizer, model) -> float:
    """Return the fineweb-edu score in [0, 5] for given text."""
    if not text or not text.strip():
        return 0.0
    try:
        # Import torch here to avoid heavy import at module load
        import torch
        
        inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        score = float(outputs.logits.squeeze(-1).float().detach().numpy().item())
        return max(0.0, min(score, 5.0))
    except Exception as e:
        logging.warning(f"calculate_edu_score failed; returning 0.0. Error: {e}", exc_info=True)
        return 0.0


def calculate_edu_score_dict(text: str, tokenizer, model) -> dict:
    """Return a dict with continuous and integer fineweb-edu scores."""
    score = calculate_edu_score(text, tokenizer, model)
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


# -----------------------------
# Pretty print utilities
# -----------------------------

def format_thinking_display(thinking_text: str) -> str:
    """
    Format thinking text for display, truncating if > 1000 characters.

    Args:
        thinking_text: The thinking content

    Returns:
        Formatted thinking text for display
    """
    if not thinking_text:
        return ""
    if len(thinking_text) <= 1000:
        return thinking_text
    first_500 = thinking_text[:500]
    last_500 = thinking_text[-500:]
    return f"{first_500}\n[...]\n{last_500}"


def format_document_structure(doc) -> str:
    """
    Format document structure without showing full content.

    Args:
        doc: Document to format structure summary for

    Returns:
        Formatted string summary of document structure
    """
    structure_info = []

    # Document ID and basic info
    if hasattr(doc, "id") and getattr(doc, "id"):
        structure_info.append(f"Document ID: {doc.id}")

    # Text length info
    text_value = getattr(doc, "text", "")
    structure_info.append(f"Document text: {len(text_value)} chars")

    # Metadata summary
    metadata = getattr(doc, "metadata", {}) or {}
    if metadata:
        structure_info.append("Metadata keys:")
        for key, value in metadata.items():
            if isinstance(value, str):
                value_summary = f"{len(value)} chars" if len(value) > 250 else f"'{value}'"
            elif isinstance(value, (int, float, bool)):
                value_summary = str(value)
            elif isinstance(value, list):
                value_summary = f"list({len(value)} items)"
            elif isinstance(value, dict):
                value_summary = f"dict({len(value)} items)"
                structure_info.append(f"  - {key}: {value_summary}")
                # Expand nested dict contents
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, str):
                        nested_summary = (
                            f"{len(nested_value)} chars" if len(nested_value) > 250 else f"'{nested_value}'"
                        )
                    elif isinstance(nested_value, (int, float, bool)):
                        nested_summary = str(nested_value)
                    elif isinstance(nested_value, (list, dict)):
                        nested_summary = f"{type(nested_value).__name__}({len(nested_value)} items)"
                    else:
                        nested_summary = f"{type(nested_value).__name__}"
                    structure_info.append(f"    - {nested_key}: {nested_summary}")
                continue  # Skip the normal append since we already added it
            else:
                value_summary = f"{type(value).__name__}"
            structure_info.append(f"  - {key}: {value_summary}")
    else:
        structure_info.append("Metadata: None")

    return "\n".join(structure_info)


def print_debug_output(document, thinking_text: str, output_text: str) -> None:
    """
    Print debug output showing input/output pair with tokens and educational scores.

    This function is resilient to missing metadata fields. If "input" or
    token/score fields are absent, it falls back to sensible defaults.

    Args:
        document: The processed (or raw) document
        thinking_text: The extracted thinking text (can be empty)
        output_text: The output text (for raw docs, pass document.text)
    """
    # Safely extract fields with fallbacks
    out_meta = getattr(document, "metadata", {}) or {}
    in_meta = out_meta.get("input", {}) if isinstance(out_meta.get("input", {}), dict) else {}
    thinking_meta = out_meta.get("thinking", {}) if isinstance(out_meta.get("thinking", {}), dict) else {}
    
    input_text = in_meta.get("text", getattr(document, "text", ""))
    input_tokens = in_meta.get("token_count", 0)
    input_edu = in_meta.get("edu_score", 0.0)
    input_dclm = in_meta.get("dclm_score", 0.0)

    thinking_tokens = thinking_meta.get("token_count", 0)
    thinking_edu = thinking_meta.get("edu_score", 0.0)
    thinking_dclm = thinking_meta.get("dclm_score", 0.0)
   
    output_tokens = out_meta.get("token_count", 0)
    output_edu = out_meta.get("edu_score", 0.0)
    output_dclm = out_meta.get("dclm_score", 0.0)
    
    document_structure = format_document_structure(document)

    log_cutoff = 2500
    delimiter_outside = "=" * 100
    delimiter_inside = "-" * 50

    thinking_part = f"""{delimiter_inside}
🧠 THINKING ({thinking_tokens} tokens, edu score: {thinking_edu:.2f}, dclm score: {thinking_dclm:.2f}):
{delimiter_inside}
{format_thinking_display(thinking_text)}
""" if thinking_tokens > 0 else ""

    print(
        f"""
{delimiter_outside}
🔍 DEBUG: INPUT/OUTPUT PAIR
{delimiter_outside}
📝 INPUT ({input_tokens} tokens, edu score: {input_edu:.2f}, dclm score: {input_dclm:.2f}):
{delimiter_inside}
{(input_text or '')[:log_cutoff] + ('...' if len(input_text or '') > log_cutoff else '')}
{thinking_part}{delimiter_inside}
🔄 OUTPUT ({output_tokens} tokens, edu score: {output_edu:.2f}, dclm score: {output_dclm:.2f}):
{delimiter_inside}
{output_text}
{delimiter_inside}
📋 DOCUMENT STRUCTURE:
{delimiter_inside}
{document_structure}
{delimiter_outside}
"""
    )


# -----------------------------
# DCLM classifier utilities (FastText OH/ELI5)
# -----------------------------

# Model hosted on Hugging Face; we store it under CACHE_PATH for reuse
_DCLM_MODEL_URL = (
    "https://huggingface.co/mlfoundations/fasttext-oh-eli5/resolve/main/"
    "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
)
_DCLM_MODEL_PATH = Path(CACHE_PATH) / "dclm" / "fasttext_oh_eli5.bin"

_DCLM_MODEL = None
_DCLM_MODEL_LOCK = threading.Lock()


def _download_dclm_model_if_needed() -> Path:
    _DCLM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _DCLM_MODEL_PATH.exists():
        logging.info(f"Downloading DCLM classifier model to {_DCLM_MODEL_PATH} ...")
        urllib.request.urlretrieve(_DCLM_MODEL_URL, _DCLM_MODEL_PATH.as_posix())
        logging.info("DCLM classifier model download complete.")
    else:
        logging.info(f"DCLM classifier model already present at {_DCLM_MODEL_PATH}.")


def _load_dclm_model():
    try:
        import fasttext  # Imported here to avoid heavy import at module load
    except Exception as e:
        logging.warning(f"Failed to import fasttext for DCLM classifier: {e}")
        raise

    _download_dclm_model_if_needed()
    logging.info(f"Loading DCLM classifier model from {_DCLM_MODEL_PATH} ...")
    model = fasttext.load_model(_DCLM_MODEL_PATH.as_posix())
    logging.info("DCLM classifier model loaded.")
    return model


def _get_dclm_model():
    global _DCLM_MODEL
    if _DCLM_MODEL is None:
        with _DCLM_MODEL_LOCK:
            if _DCLM_MODEL is None:
                _DCLM_MODEL = _load_dclm_model()
    return _DCLM_MODEL


def calculate_dclm_score(text: str) -> float:
    """Return the DCLM classifier score (probability of __label__hq) for given text.

    Find more information about the classifier here: https://github.com/mlfoundations/dclm/tree/main/baselines#fasttext-filtering

    Returns a float in [0, 1]. On error or empty text, returns 0.0.
    """
    if not text or not text.strip():
        return 0.0
    try:
        threshold = 0.018112 # default
        # FastText expects a single line of text; remove newlines and collapse whitespace
        sanitized_text = " ".join(text.split())
        if not sanitized_text:
            return 0.0
        model = _get_dclm_model()
        labels, probs = model.predict(sanitized_text, k=2, threshold=threshold)
        label_to_prob = {label: prob for label, prob in zip(labels, probs)}
        return float(label_to_prob.get("__label__hq", 0.0))
    except Exception as e:
        logging.warning(f"calculate_dclm_score failed; returning 0.0. Error: {e}", exc_info=True)
        return 0.0

