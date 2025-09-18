"""
Document rephrasing pipeline with comprehensive quality assessment.

This script processes documents through an LLM to rephrase/rewrite them using 
customizable prompt templates. It extracts thinking and final output, calculates 
educational quality scores with fineweb-edu-classifier, counts tokens, and 
organizes metadata hierarchically. Supports chunked processing with checkpoints, 
debug output, and stores complete processing provenance.
"""

import argparse
import os
from typing import Any

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner

from utils import (
    FAULTY_NODES,
    LOCAL_TMP_PATH_ON_NODE,
    LOG_BASE_PATH,
    S3_BASE_PATH,
    build_reader,
    EduScoreStatsLogger,
    calculate_edu_score_dict,
)


def load_prompt_template(template_path: str) -> str:
    """
    Load a prompt template from a file in the prompts directory.
    
    Args:
        template_path:  Path to template file relative to prompts/ directory
        
    Returns:
        Template content as a string
    """
    if not template_path:
        return None
        
    # Get the base directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, "prompts", template_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Prompt template not found: {full_path}")
        
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def create_templated_query_builder(
    system_prompt: str,
    max_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    enable_thinking: bool = False,
):
    """
    Create a query builder function that uses a system prompt (as a system message)
    and a user prompt (as a user message) formatted with the document text.
    
    Args:
        system_prompt:          The system prompt content
        max_tokens:             Maximum tokens for the response
        temperature:            Temperature for inference
        top_p:                  Top-p (nucleus sampling) for inference
        top_k:                  Top-k sampling for inference
        enable_thinking:        Enable thinking in chat template
        
    Returns:
        Query builder function
    """

    assert system_prompt, "System prompt is required"

    # Default user prompt if none is provided
    user_prompt = "[[ ## original_text ## ]]\n[TEXT]\n\n"
    user_prompt += "Respond with the corresponding output fields, starting with the field `[[ ## generated_text ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."

    def query_builder(runner: InferenceRunner, document: Document) -> dict[str, Any]:
        """
        Query builder that applies the system and user templates to construct chat messages.
        
        Args:
            runner:     Inference runner instance
            document:   Input document with text content
            
        Returns:
            Query payload for the inference server
        """
        # Prepare user content by replacing common placeholders
        user_prompt = load_prompt_template("dspy/user.md")
        user_content = user_prompt.replace("[DOCUMENT SEGMENT]", document.text)
        user_content = user_content.replace("[ORIGINAL DOCUMENT]", document.text)
        user_content = user_content.replace("[TEXT]", document.text)

        # Truncate user content if too long to avoid server errors
        max_chars = 4 * max_tokens  # rough heuristic for average token length
        max_chars *= 2 # Since our model_max_length is 16384 and our max_tokens is 4096, we can afford to use more input tokens
        if len(user_content) > max_chars:
            cutoff_content = user_content[:max_chars]
            last_newline = cutoff_content.rfind('\n')
            if last_newline != -1:
                user_content = user_content[:last_newline]
            else:
                user_content = user_content[:max_chars]

        return {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    return query_builder


def create_postprocess_fn(debug: bool = False, tokenizer_name=None, model_name=None):
    """
    Create a postprocess function that parses thinking/output and saves data to metadata.
    
    Args:
        debug: Whether to enable debug logging to terminal
        tokenizer_name: Name of the tokenizer used
        model_name: Name of the rephrasing model used
    
    Returns:
        Postprocess function that processes LLM output and saves to metadata
    """
    # Lazily initialized cache for token counting tokenizer
    tokenizer = None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    edu_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
    edu_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier").eval()
        

    def count_tokens(text: str) -> int:
        """Count tokens in text using tokenizer; lazily load on first use."""
        nonlocal tokenizer
        if text:
            if tokenizer is None and tokenizer_name:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer:
                return len(tokenizer.encode(text))
        return 0
    

    def parse_thinking_output(text: str) -> tuple[str, str]:
        """
        Parse output text into thinking and final output parts.
        
        Args:
            text: Raw output text that may contain <think></think> tags
            
        Returns:
            Tuple of (thinking_text, final_output_text)
        """
        thinking_text = ""
        final_output_text = text
        
        # Look for <think> and </think> tags
        if "<think>" in text and "</think>" in text:
            think_start = text.find("<think>")
            think_end = text.find("</think>", think_start)
            
            if think_start != -1 and think_end != -1:
                # Extract thinking content (without tags)
                thinking_text = text[think_start + 7:think_end].strip()
                
                # Remove thinking section from final output
                final_output_text = text[:think_start] + text[think_end + 8:]
                final_output_text = final_output_text.strip()
        
        return thinking_text, final_output_text

    def extract_text_from_inference_results(inference_results: list) -> str:
        """
        Extract text from inference results, handling various result formats.
        
        Args:
            inference_results: List of inference results from the inference pipeline
            
        Returns:
            Extracted text string, or empty string if no text found
        """
        if not inference_results:
            return ""
            
        # Get the first successful result
        for result in inference_results:
            if hasattr(result, 'text'):  # InferenceSuccess object
                return result.text
            elif isinstance(result, dict):
                # Extract from dict format
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"]
                    elif "text" in choice:
                        return choice["text"]
                elif "content" in result:
                    return result["content"]
                elif "text" in result:
                    return result["text"]
        
        return ""
    
    def parse_structured_generated_text(text: str) -> str:
        """
        Extract the generated text from the structured output format:
        [[ ## generated_text ## ]]
        {text}
        [[ ## completed ## ]]

        If markers are missing, fall back to the full text.
        """
        if not text:
            return ""
        start_marker = "[[ ## generated_text ## ]]"
        end_marker = "[[ ## completed ## ]]"
        start_idx = text.find(start_marker)
        if start_idx == -1:
            return text.strip()
        start_idx += len(start_marker)
        # Skip a potential leading newline
        if start_idx < len(text) and text[start_idx] == "\n":
            start_idx += 1
        end_idx = text.find(end_marker, start_idx)
        if end_idx == -1:
            end_idx = len(text)
        return text[start_idx:end_idx].strip()
    
    def format_thinking_display(thinking_text: str) -> str:
        """
        Format thinking text for display, truncating if > 1000 characters.
        
        Args:
            thinking_text: The thinking content
            
        Returns:
            Formatted thinking text for display
        """
        if len(thinking_text) <= 1000:
            return thinking_text
        else:
            first_500 = thinking_text[:500]
            last_500 = thinking_text[-500:]
            return f"{first_500}\n[...]\n{last_500}"
    
    def format_document_structure(doc: Document) -> str:
        """
        Format document structure without showing full content.
        
        Args:
            doc: Document to format structure summary for
            
        Returns:
            Formatted string summary of document structure
        """
        structure_info = []
        
        # Document ID and basic info
        if hasattr(doc, 'id') and doc.id:
            structure_info.append(f"Document ID: {doc.id}")
        
        # Text length info
        structure_info.append(f"Document text: {len(doc.text)} chars")
        
        # Metadata summary
        if doc.metadata:
            structure_info.append("Metadata keys:")
            for key, value in doc.metadata.items():
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
                            nested_summary = f"{len(nested_value)} chars" if len(nested_value) > 250 else f"'{nested_value}'"
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
    
    def print_debug_output(document: Document, thinking_text: str, final_output_text: str) -> None:
        """
        Print debug output showing input/output pair with tokens and educational scores.
        
        Args:
            document: The processed document with metadata
            thinking_text: The extracted thinking text
            final_output_text: The final output text
        """
        # Get tokens from metadata
        input_tokens = document.metadata["input"]["token_count"]
        thinking_tokens = document.metadata["thinking"]["token_count"]
        final_output_tokens = document.metadata["token_count"]
        
        document_structure = format_document_structure(document)

        log_cutoff = 2500
        delimiter_outside = '='*100
        delimiter_inside = '-'*50
        
        print(f"""
{delimiter_outside}
🔍 DEBUG: INPUT/OUTPUT PAIR
{delimiter_outside}
📝 INPUT ({input_tokens} tokens, edu score: {document.metadata["input"]["score"]:.2f}/{document.metadata["input"]["int_score"]}):
{delimiter_inside}
{document.metadata["input"]["text"][:log_cutoff] + ("..." if len(document.metadata["input"]["text"]) > log_cutoff else "")}
{delimiter_inside}
🧠 THINKING ({thinking_tokens} tokens, edu score: {document.metadata["thinking"]["score"]:.2f}/{document.metadata["thinking"]["int_score"]}):
{delimiter_inside}
{format_thinking_display(thinking_text)}
{delimiter_inside}
🔄 FINAL OUTPUT ({final_output_tokens} tokens, edu score: {document.metadata["score"]:.2f}/{document.metadata["int_score"]}):
{delimiter_inside}
{final_output_text}
{delimiter_inside}
📋 DOCUMENT STRUCTURE:
{delimiter_inside}
{document_structure}
{delimiter_outside}
""")
            
    def postprocess_fn(document: Document) -> Document:
        """
        Postprocess function that parses thinking/output and saves data to metadata.
        Sets document.text to the final output.
        
        Args:
            document: Document with inference results in metadata
            
        Returns:
            The document with updated text and metadata
        """
        # Extract inference results from document metadata
        inference_results = document.metadata.get("inference_results", [])
        
        output_text = extract_text_from_inference_results(inference_results)
        
        # Parse thinking and final output
        thinking_text, final_output_raw = parse_thinking_output(output_text)
        # Extract the generated text segment if present
        final_output_text = parse_structured_generated_text(final_output_raw)
        
        # Calculate metrics for input, thinking, and final output
        input_token_count = count_tokens(document.text)
        thinking_token_count = count_tokens(thinking_text)
        final_output_token_count = count_tokens(final_output_text)
        
        input_edu_scores = calculate_edu_score_dict(document.text, edu_tokenizer, edu_model)
        thinking_edu_scores = calculate_edu_score_dict(thinking_text, edu_tokenizer, edu_model)
        final_output_edu_scores = calculate_edu_score_dict(final_output_text, edu_tokenizer, edu_model)
        
        # Collect input-related metadata fields to move
        input_metadata_fields = ["dump", "url", "date", "file_path", "language", "language_score", "filter_reason"]
        input_metadata = {}
        for field in input_metadata_fields:
            if field in document.metadata:
                input_metadata[field] = document.metadata.pop(field)
        
        # Store data in hierarchical metadata structure
        document.metadata["input"] = {
            "text": document.text,
            "token_count": input_token_count,
            **input_edu_scores,
            **input_metadata
        }
        
        document.metadata["thinking"] = {
            "text": thinking_text,
            "token_count": thinking_token_count,
            **thinking_edu_scores
        }
        
        # Store final output metrics at top level
        document.metadata["token_count"] = final_output_token_count
        document.metadata.update(final_output_edu_scores)
        document.metadata["token_reduction"] = input_token_count - final_output_token_count
        document.metadata["edu_score_difference"] = final_output_edu_scores["score"] - input_edu_scores["score"]
        document.metadata["edu_score_improvement"] = 1 if final_output_edu_scores["score"] > input_edu_scores["score"] else 0
        
        # Store processing configuration
        if tokenizer_name:
            document.metadata["tokenizer_name"] = tokenizer_name
        if model_name:
            document.metadata["rephrasing_model_name"] = model_name
        
        # Set document text to the final output
        document.text = final_output_text
        
        # Debug output (only if debug flag is enabled)
        if debug:
            print_debug_output(document, thinking_text, final_output_text)
        
        return document
    
    return postprocess_fn


# Create argument parser
parser = argparse.ArgumentParser("Rephrase documents using inference pipeline.")

parser.add_argument(
    "--data-paths", type=str, help="Path to the data to rephrase.", required=True
)
parser.add_argument(
    "--name", type=str, help="Name of the rephrasing experiment", required=True
)
parser.add_argument(
    "--prompt", type=str, help="Path to prompt template file (relative to prompts/ directory)", required=True
)
parser.add_argument(
    "--output-path", type=str, help="Path to the base output folder.", default=S3_BASE_PATH
)
parser.add_argument(
    "--model-name-or-path", type=str, help="Model name or path for inference", default="google/gemma-3-270m-it"
)
parser.add_argument(
    "--limit", type=int, help="Limit the number of documents to rephrase", default=-1
)
parser.add_argument(
    "--n-tasks", type=int, help="Number of parallel tasks", default=1
)
parser.add_argument(
    "--temperature", type=float, help="Temperature for inference", default=1
)
parser.add_argument(
    "--top-p", type=float, help="Top-p (nucleus sampling) for inference", default=0.95
)
parser.add_argument(
    "--top-k", type=int, help="Top-k sampling for inference", default=64
)
parser.add_argument(
    "--enable-thinking", action="store_true", help="Enable thinking in chat template"
)
parser.add_argument(
    # Allow for long inputs to account for the prompt and because the model may output shorter rephrases
    "--model-max-context", type=int, help="Maximum context length for the model", default=16384
)
parser.add_argument(
    "--max-concurrent-requests", type=int, help="Maximum concurrent requests", default=500
)
parser.add_argument(
    "--max-concurrent-tasks", type=int, help="Maximum concurrent tasks", default=500
)
parser.add_argument(
    "--metric-interval", type=int, help="Metric logging interval in seconds", default=60
)
parser.add_argument(
    "--records-per-chunk", type=int, help="Number of records per chunk", default=1000
)
parser.add_argument(
    # We only train on this many tokens, so no need to go beyond
    "--max-tokens", type=int, help="Maximum tokens per request", default=4096 
)
parser.add_argument(
    "--server-type", type=str, help="Inference server type", choices=["vllm", "sglang", "dummy"], default="vllm"
)
parser.add_argument(
    "--text-key", type=str, default="text", help="Key name for text content in input documents"
)
parser.add_argument(
    "--disable-checkpoints", action="store_true", help="Disable checkpoint functionality"
)
parser.add_argument(
    "--debug", action="store_true", help="Enable debug logging to show input/output text pairs in terminal (overrides limit, run_local and disable_checkpoints)"
)
parser.add_argument(
    "--tokenizer", type=str, default="hynky/Llama-3.2-1B-no-bos", help="Tokenizer to use for token counting in debug output"
)
parser.add_argument(
    "--time", type=str, default="20:00:00", help="Slurm time limit"
)
parser.add_argument(
    "--qos", type=str, default="normal", help="Slurm QoS"
)
parser.add_argument(
    "--dep-job-id", type=str, default=None, help="Optional Slurm dependency job id"
)
parser.add_argument(
    "--gpus", type=int, default=1, help="Number of GPUs per task"
)
parser.add_argument(
    "--enable-prefix-caching", action="store_true", default=True, help="Enable prefix caching"
)
parser.add_argument(
    "--enable-chunked-prefill", action="store_true", default=True, help="Enable chunked prefill"
)
parser.add_argument(
    "--tp", type=int, default=1, help="Tensor parallelism size"
)
parser.add_argument(
    "--gpu-memory-utilization", type=float, default=0.97, help="GPU memory utilization"
)
parser.add_argument(
    "--run-local", action="store_true", help="Run pipeline locally instead of using Slurm"
)


def main():
    args = parser.parse_args()

    skip_completed = True
    if args.debug:
        args.run_local = True
        args.disable_checkpoints = True
        args.limit = 3
        skip_completed = False
        print("🔍 DEBUG MODE ENABLED: Input/output pairs will be logged to terminal")
    else:
        # Safety guard: Abort submission for unsafe configurations (skip in debug/local)
        if args.qos.lower() in {"normal", "high"}:
            if args.limit == -1 or args.limit > 10000 or args.n_tasks > 10:
                print(f"It looks like you are trying to run a large rephrasing experiment. " \
                "Please change qos to low and inform people in #science-cluster-planning about large runs.")
                raise ValueError("Unsafe configuration")
    
    # Set up paths based on arguments
    run_name = f"{args.prompt.replace('.md', '')}-{args.model_name_or_path.split('/')[-1]}-{args.name}"
    output_path = f"{args.output_path}/rephrased/{run_name}"
    logs_path = f"{LOG_BASE_PATH}/rephrasing/{run_name}"
    checkpoints_path = f"{LOCAL_TMP_PATH_ON_NODE}/checkpoints/{run_name}" if not args.disable_checkpoints else None

    # Parse data paths
    data_paths = args.data_paths.split(",")
    print(f"Data paths: {data_paths}")
    print(f"Output path: {output_path}")

    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.executor.slurm import SlurmPipelineExecutor
    from datatrove.executor.local import LocalPipelineExecutor

    # Build model kwargs for the inference server
    _model_kwargs = {
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "bfloat16",
    }

    config: InferenceConfig = InferenceConfig(
        server_type=args.server_type,
        model_name_or_path=args.model_name_or_path,
        temperature=args.temperature,
        model_max_context=args.model_max_context,
        max_concurrent_requests=args.max_concurrent_requests,
        max_concurrent_tasks=args.max_concurrent_tasks,
        metric_interval=args.metric_interval,
        tp=args.tp,
        model_kwargs=_model_kwargs,
    )

    # Load prompt template if specified
    try:
        prompt_template = load_prompt_template(args.prompt)
        print(f"Loaded prompt template: {args.prompt}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Available templates:")
        prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        for root, _, files in os.walk(prompts_dir):
            for file in files:
                if file.endswith('.md'):
                    rel_path = os.path.relpath(os.path.join(root, file), prompts_dir)
                    print(f"  - {rel_path}")
        exit(1)

    pipeline = [
            *[build_reader(data_path, limit=args.limit, n_tasks=args.n_tasks, shuffle_files=False, text_key=args.text_key) for data_path in data_paths],
            InferenceRunner(
                query_builder=create_templated_query_builder(
                    prompt_template, 
                    args.max_tokens, 
                    args.temperature, 
                    args.top_p, 
                    args.top_k, 
                    args.enable_thinking
                ),
                config=config,
                records_per_chunk=args.records_per_chunk,
                checkpoints_local_dir=checkpoints_path,
                output_writer=JsonlWriter(output_path, output_filename="${rank}_chunk_${chunk_index}.jsonl"),
                skip_bad_requests=True, # Skips documents that cause BadRequestError from the server, e.g., when they are too long
                postprocess_fn=create_postprocess_fn(
                    debug=args.debug,
                    tokenizer_name=args.tokenizer,
                    model_name=args.model_name_or_path
                ),
            ),
            JsonlReader(output_path),
            EduScoreStatsLogger(),
        ]

    if args.run_local:
        rephrase_executor = LocalPipelineExecutor(
            pipeline=pipeline, logging_dir=logs_path, skip_completed=skip_completed
        )
    else:
        rephrase_executor = SlurmPipelineExecutor(
            job_name=f"rephrase-{run_name}",
            pipeline=pipeline,
            logging_dir=logs_path,
            tasks=args.n_tasks,
            time=args.time,
            partition="hopper-prod",
            cpus_per_task=10*args.gpus,
            mem_per_cpu_gb=20,
            qos=args.qos,
            env_command="sleep $((RANDOM % 30))",
            mail_user="joel@hf.co",
            depends_job_id=args.dep_job_id,
            sbatch_args={"gres": f"gpu:{args.gpus}", "exclude": FAULTY_NODES}
        )
    
    rephrase_executor.run()
    

if __name__ == "__main__":
    main()
