import os
import logging
from datatrove.pipeline.base import PipelineStep

from dotenv import load_dotenv

load_dotenv()

USER = os.environ.get('USER')
PROJECT_NAME = "finephrase"

BASE_PATH = f"/fsx/{USER}"
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


def print_debug_output(document, thinking_text: str, final_output_text: str) -> None:
    """
    Print debug output showing input/output pair with tokens and educational scores.

    This function is resilient to missing metadata fields. If "input" or
    token/score fields are absent, it falls back to sensible defaults.

    Args:
        document: The processed (or raw) document
        thinking_text: The extracted thinking text (can be empty)
        final_output_text: The final output text (for raw docs, pass document.text)
    """
    # Safely extract fields with fallbacks
    metadata = getattr(document, "metadata", {}) or {}
    input_meta = metadata.get("input", {}) if isinstance(metadata.get("input", {}), dict) else {}
    input_text = input_meta.get("text", getattr(document, "text", ""))
    input_tokens = input_meta.get("token_count", 0)

    thinking_meta = metadata.get("thinking", {}) if isinstance(metadata.get("thinking", {}), dict) else {}
    thinking_tokens = thinking_meta.get("token_count", 0)

    final_output_tokens = metadata.get("token_count", 0)
    output_score = metadata.get("score", 0.0)
    output_int_score = metadata.get("int_score", 0)

    input_scores = {
        "score": input_meta.get("score", 0.0),
        "int_score": input_meta.get("int_score", 0),
    }
    thinking_scores = {
        "score": thinking_meta.get("score", 0.0),
        "int_score": thinking_meta.get("int_score", 0),
    }

    document_structure = format_document_structure(document)

    log_cutoff = 2500
    delimiter_outside = "=" * 100
    delimiter_inside = "-" * 50

    thinking_part = f"""{delimiter_inside}
🧠 THINKING ({thinking_tokens} tokens, edu score: {thinking_scores['score']:.2f}/{thinking_scores['int_score']}):
{delimiter_inside}
{format_thinking_display(thinking_text)}
""" if thinking_tokens > 0 else ""

    print(
        f"""
{delimiter_outside}
🔍 DEBUG: INPUT/OUTPUT PAIR
{delimiter_outside}
📝 INPUT ({input_tokens} tokens, edu score: {input_scores['score']:.2f}/{input_scores['int_score']}):
{delimiter_inside}
{(input_text or '')[:log_cutoff] + ('...' if len(input_text or '') > log_cutoff else '')}
{thinking_part}{delimiter_inside}
🔄 FINAL OUTPUT ({final_output_tokens} tokens, edu score: {output_score:.2f}/{output_int_score}):
{delimiter_inside}
{final_output_text}
{delimiter_inside}
📋 DOCUMENT STRUCTURE:
{delimiter_inside}
{document_structure}
{delimiter_outside}
"""
    )