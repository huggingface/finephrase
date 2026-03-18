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

from transformers import GenerationConfig


from utils import (
    CHECKPOINTS_PATH,
    ENV_COMMAND,
    FAULTY_NODES,
    LOG_BASE_PATH,
    S3_BASE_PATH,
    build_reader,
    print_debug_output,
)
from quality_scores import (
    QualityScoreStatsLogger,
    calculate_edu_score,
    calculate_dclm_score,
)

def load_prompt_template(template_path: str) -> str:
    """
    Load a prompt template from a file in the prompts directory.
    
    Args:
        template_path:  Path to template file relative to prompts/ directory
        
    Returns:
        Template content as a string
    """
     
    # Get the base directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, "prompts", template_path)
    
    if not template_path or not os.path.exists(full_path):
        raise FileNotFoundError(f"Prompt template not found: {full_path}")
        
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def create_templated_query_builder(
    prompt_path: str,
    max_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    enable_thinking: bool = False,
    model_max_context: int | None = None,
):
    """
    Create a query builder function that loads and applies the prompt template.
    
    Args:
        prompt_path:            Path to the prompt template
        max_tokens:             Maximum tokens for the response
        temperature:            Temperature for inference
        top_p:                  Top-p (nucleus sampling) for inference
        top_k:                  Top-k sampling for inference
        enable_thinking:        Enable thinking in chat template
        model_max_context:      Model maximum context window in tokens
        
    Returns:
        Query builder function
    """
    
    is_dspy = prompt_path.startswith("dspy")
    prompt_template = load_prompt_template(prompt_path)

    def query_builder(runner: InferenceRunner, document: Document) -> dict[str, Any]:
        """
        Query builder that applies the prompt template to construct chat messages.
        
        Args:
            runner:     Inference runner instance
            document:   Input document with text content
            
        Returns:
            Query payload for the inference server
        """
        # Prepare user content
        user_content = load_prompt_template("dspy/user.md") if is_dspy else prompt_template
        user_content = user_content.replace("[TEXT]", document.text)

        # Truncate user content if too long to avoid server errors
        char_budget = (model_max_context - max_tokens) * 4

        if len(user_content) > char_budget:
            last_newline = user_content.rfind('\n', 0, char_budget)
            user_content = user_content[:last_newline] if last_newline != -1 else user_content[:char_budget]

        messages = []
        
        # Only add system message for dspy prompts
        if is_dspy:
            messages.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": prompt_template},
                ],
            })
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_content},
            ],
        })

        return {
            "messages": messages,
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
        

    def count_tokens(text: str) -> int:
        """Count tokens in text using tokenizer; lazily load on first use."""
        nonlocal tokenizer
        if text:
            if tokenizer is None and tokenizer_name:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
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
        Also removes "Here is a paraphrased version:" prefix if present.
        """
        if not text:
            return ""
        start_marker = "[[ ## generated_text ## ]]"
        end_marker = "[[ ## completed ## ]]"
        start_idx = text.find(start_marker)
        if start_idx == -1:
            result = text.strip()
        else:
            start_idx += len(start_marker)
            # Skip a potential leading newline
            if start_idx < len(text) and text[start_idx] == "\n":
                start_idx += 1
            end_idx = text.find(end_marker, start_idx)
            if end_idx == -1:
                end_idx = len(text)
            result = text[start_idx:end_idx].strip()
        
        # Remove "Here is a paraphrased version:" prefix if present
        prefix = "Here is a paraphrased version:"
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
        
        return result
    
            
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
        
        # Calculate EDU scores
        input_edu_score = calculate_edu_score(document.text)
        thinking_edu_score = calculate_edu_score(thinking_text)
        final_output_edu_score = calculate_edu_score(final_output_text)

        # Calculate DCLM scores
        input_dclm_score = calculate_dclm_score(document.text)
        thinking_dclm_score = calculate_dclm_score(thinking_text)
        final_output_dclm_score = calculate_dclm_score(final_output_text)
        
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
            "edu_score": input_edu_score,
            "dclm_score": input_dclm_score,
            **input_metadata
        }
        
        document.metadata["thinking"] = {
            "text": thinking_text,
            "token_count": thinking_token_count,
            "edu_score": thinking_edu_score,
            "dclm_score": thinking_dclm_score
        }
        
        # Store final output metrics at top level
        document.metadata["token_count"] = final_output_token_count
        document.metadata["edu_score"] = final_output_edu_score
        document.metadata["dclm_score"] = final_output_dclm_score
        document.metadata["token_reduction"] = input_token_count - final_output_token_count
        document.metadata["edu_score_difference"] = final_output_edu_score - input_edu_score
        document.metadata["edu_score_improvement"] = 1 if final_output_edu_score > input_edu_score else 0
        document.metadata["dclm_score_difference"] = final_output_dclm_score - input_dclm_score
        document.metadata["dclm_score_improvement"] = 1 if final_output_dclm_score > input_dclm_score else 0
        
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
    "--data", type=str, help="Path to the data to rephrase.", required=True
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
    "--model-name-or-path", type=str, help="Model name or path for inference", default="google/gemma-3-4b-it"
)
parser.add_argument(
    "--limit", type=int, help="Limit the number of documents to rephrase", default=-1
)
parser.add_argument(
    "--n-tasks", type=int, help="Number of parallel tasks", default=1000
)
parser.add_argument(
    "--n-workers", type=int, help="Number of workers (jobs to run in parallel)", default=-1
)
parser.add_argument(
    "--temperature", type=float, help="Temperature for inference", default=None
)
parser.add_argument(
    "--top-p", type=float, help="Top-p (nucleus sampling) for inference", default=None
)
parser.add_argument(
    "--top-k", type=int, help="Top-k sampling for inference", default=None
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
    "--time", type=str, default="3-00:00:00", help="Slurm time limit"
)
parser.add_argument(
    "--qos", type=str, default="low", help="Slurm QoS"
)
parser.add_argument(
    "--override-unsafe", action="store_true", help="Override safety guard for large runs"
)
parser.add_argument(
    "--dep-job-id", type=str, default=None, help="Optional Slurm dependency job id"
)
parser.add_argument(
    "--enable-prefix-caching", action="store_true", default=False, help="Enable prefix caching"
)
parser.add_argument(
    "--enable-chunked-prefill", action="store_true", default=True, help="Enable chunked prefill"
)
parser.add_argument(
    "--tp", type=int, default=1, help="Tensor parallelism size"
)
parser.add_argument(
    "--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization" # without speculative decoding we can use 0.97
)
parser.add_argument(
    "--speculative-config", type=str, default='{"method": "ngram", "num_speculative_tokens": 8, "prompt_lookup_max": 7}', help="Speculative decoding configuration"
)
parser.add_argument(
    "--run-local", action="store_true", help="Run pipeline locally instead of using Slurm"
)
parser.add_argument(
    "--reservation", type=str, default=None, help="Slurm reservation"
)


def main():
    args = parser.parse_args()

    if args.debug:
        args.run_local = True
        args.disable_checkpoints = True
        args.n_tasks = 1
        args.limit = 5
        print("🔍 DEBUG MODE ENABLED: Input/output pairs will be logged to terminal")
        
        # Check if GPUs are available in debug mode
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                pass  # GPUs available, continue
            else:
                raise RuntimeError("nvidia-smi failed")
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError):
            print(f"\n❌ No GPUs available. Please run:")
            print(f'srun --gpus={args.tp} --qos=high --time="04:00:00" --pty bash')
            print("\nThen run this script again.")
            exit(1)
    else:
        print("⚠️ Please inform people in #science-cluster-planning about large runs.")
        # Safety guard: Abort submission for unsafe configurations (skip in debug/local)
        if args.qos.lower() in {"normal", "high"} \
            and (args.limit == -1 or args.limit > 10000) \
            and (args.n_workers == -1 or args.n_workers > 16):
            print(f"It looks like you are trying to run a large rephrasing experiment. " \
            "Please change qos to low or limit the number of workers.")
            if not args.override_unsafe:
                raise ValueError("Unsafe configuration")
            else:
                print("Proceeding despite unsafe configuration due to --override-unsafe.")

    # Pre-cache models before job submission to ensure they're available when HF_HUB_OFFLINE=1 is set on workers
    # This runs in the main submission process where network access is available
    if not args.run_local:
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        print("Verifying models are cached (required for offline workers)...")
        try:
            # Cache the full inference model (used by vLLM) - downloads all files without loading into memory
            print(f"  - Caching {args.model_name_or_path} (model + tokenizer)...")
            snapshot_download(repo_id=args.model_name_or_path, ignore_patterns=["*.gguf", "*.msgpack"])
            
            # Cache the token counting tokenizer
            print(f"  - Caching {args.tokenizer} tokenizer...")
            AutoTokenizer.from_pretrained(args.tokenizer)
            
            print("✓ All models cached successfully - workers can run offline")
        except Exception as e:
            print(f"\n❌ ERROR: Failed to cache models: {e}")
            print("Workers will fail with HF_HUB_OFFLINE=1 if models aren't cached.")
            print("Please ensure models are downloaded before submitting jobs.\n")
            raise

    # Set up paths based on arguments
    run_name = f"{args.prompt.replace('.md', '')}-{args.name}"
    output_path = f"{args.output_path}/rephrased/{run_name}"
    logs_path = f"{LOG_BASE_PATH}/rephrasing/{run_name}"

    # Parse data paths
    data_paths = args.data.split(",")
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
        **({"speculative_config": args.speculative_config} if args.speculative_config else {}),
    }

    generation_config = GenerationConfig.from_pretrained(args.model_name_or_path, local_files_only=True)
    args.temperature=args.temperature if args.temperature is not None else getattr(generation_config, "temperature", 1.0)
    args.top_p=args.top_p if args.top_p is not None else getattr(generation_config, "top_p", 1.0)
    args.top_k=args.top_k if args.top_k is not None else getattr(generation_config, "top_k", -1)

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
        server_log_folder=logs_path + "/server_logs",
    )


    pipeline = [
            *[build_reader(data_path, limit=args.limit, n_tasks=args.n_tasks, shuffle_files=False, text_key=args.text_key) for data_path in data_paths],
            InferenceRunner(
                query_builder=create_templated_query_builder(
                    args.prompt,
                    args.max_tokens, 
                    args.temperature, 
                    args.top_p, 
                    args.top_k, 
                    args.enable_thinking,
                    model_max_context=args.model_max_context,
                ),
                config=config,
                records_per_chunk=1000,
                checkpoints_local_dir=f"{CHECKPOINTS_PATH}/{run_name}" if not args.disable_checkpoints else None, # Cannot be on scratch because it needs to be available everywhere!
                output_writer=JsonlWriter(output_path, output_filename="${rank}_chunk_${chunk_index}.jsonl"),
                skip_bad_requests=True, # Skips documents that cause BadRequestError from the server, e.g., when they are too long
                postprocess_fn=create_postprocess_fn(
                    debug=args.debug,
                    tokenizer_name=args.tokenizer,
                    model_name=args.model_name_or_path,
                ),
            ),
            JsonlReader(output_path),
            QualityScoreStatsLogger(),
        ]

    if args.run_local:
        rephrase_executor = LocalPipelineExecutor(
            pipeline=pipeline, logging_dir=logs_path, skip_completed=not args.debug
        )
    else:
        rephrase_executor = SlurmPipelineExecutor(
            job_name=f"rephrase-{run_name}",
            pipeline=pipeline,
            logging_dir=logs_path,
            tasks=args.n_tasks,
            workers=args.n_workers, # when I want to run something in qos normal, I can limit the number of GPUs
            time=args.time,
            partition="hopper-prod",
            cpus_per_task=11*args.tp,
            mem_per_cpu_gb=22,
            qos=args.qos,
            # Add the HF_HUB_OFFLINE=1 command to prevent continuous Hub requests from workers (they should use cached models only)
            env_command=ENV_COMMAND + " && export HF_HUB_OFFLINE=1",
            mail_user="joel@hf.co",
            depends_job_id=args.dep_job_id,
            sbatch_args={
                "gres": f"gpu:{args.tp}",
                **({"reservation": args.reservation} if args.reservation else {}),
            },
        )
    
    rephrase_executor.run()
    

if __name__ == "__main__":
    main()
