import os
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
    "ip-26-0-163-58",
    "ip-26-0-164-0",
    "ip-26-0-164-25",
    "ip-26-0-164-236",
    "ip-26-0-166-125",
    "ip-26-0-166-214",
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
        text_key: Text field key for JSONL inputs (ignored by parquet). Can be a comma-separated list to concatenate.

    Returns:
        An initialized reader instance
    """
    per_task_limit = -1 if limit < 0 else limit // n_tasks
    # If multiple text keys are provided as a comma-separated list, build a custom adapter
    adapter = None
    primary_text_key = text_key
    if isinstance(text_key, str) and ("," in text_key):
        concat_keys = [k.strip() for k in text_key.split(",") if k.strip()]
        if len(concat_keys) >= 2:
            primary_text_key = concat_keys[0]

            def _concat_adapter(self, data: dict, path: str, id_in_file: int | str):
                # Mirror BaseReader._default_adapter behavior while concatenating multiple fields
                metadata = data.pop("metadata", {})
                if isinstance(metadata, str):
                    import json
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        pass
                if not isinstance(metadata, dict):
                    metadata = {"metadata": metadata}

                parts = []
                for key in concat_keys:
                    value = data.pop(key, "")
                    if value is None:
                        value = ""
                    elif not isinstance(value, str):
                        value = str(value)
                    if value:
                        parts.append(value)
                text = "\n\n".join(parts)

                return {
                    "text": text,
                    "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
                    "media": data.pop("media", []),
                    "metadata": metadata | data,
                }
            adapter = _concat_adapter
    # Restrict files to appropriate types by default to avoid attempting to read non-data files
    if path.startswith("hf://"):
        glob_pattern = "**/*.parquet"
    elif path.startswith("s3://"):
        glob_pattern = "**/*.jsonl"
    else:
        raise ValueError(f"Invalid path: {path}")
    return get_reader(path)(
        path,
        shuffle_files=shuffle_files,
        limit=per_task_limit,
        adapter=adapter,
        text_key=primary_text_key,
        glob_pattern=glob_pattern,
    )


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


