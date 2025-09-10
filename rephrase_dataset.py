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

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.base import PipelineStep
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor

from utils import LOCAL_TMP_PATH_ON_NODE, LOG_BASE_PATH, S3_BASE_PATH

class EduScoreStatsLogger(PipelineStep):
    """
    Pipeline step that logs education score statistics from document metadata.
    """
    
    type = "📊 - STATS"
    name = "Education Score Stats Logger"
    
    def __init__(self):
        super().__init__()
    
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
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


def create_templated_query_builder(prompt_template: str, max_tokens: int, temperature: float = 0.7, top_p: float = 0.8, top_k: int = 20, presence_penalty: float = 1.5, enable_thinking: bool = False):
    """
    Create a query builder function that uses a prompt template.
    
    Args:
        prompt_template:    The prompt template content with placeholders
        max_tokens:         Maximum tokens for the response
        temperature:        Temperature for inference
        top_p:              Top-p (nucleus sampling) for inference
        top_k:              Top-k sampling for inference
        presence_penalty:   Presence penalty for inference
        enable_thinking:    Enable thinking in chat template
        
    Returns:
        Query builder function
    """

    assert prompt_template, "Prompt template is required"

    def query_builder(runner: InferenceRunner, document: Document) -> dict[str, Any]:
        """
        Query builder that applies a prompt template to document content.
        
        Args:
            runner:     Inference runner instance
            document:   Input document with text content
            
        Returns:
            Query payload for the inference server
        """
        # Replace common placeholders in the template
        content = prompt_template
        content = content.replace("[DOCUMENT SEGMENT]", document.text)
        content = content.replace("[ORIGINAL DOCUMENT]", document.text)
        content = content.replace("[TEXT]", document.text)
        
        # Truncate if content is too long to make sure the server doesn't throw an error
        max_chars = 4 * max_tokens # rough heuristic for average token length
        if len(content) > max_chars:
            # Find the last newline before the cutoff
            cutoff_content = content[:max_chars]
            last_newline = cutoff_content.rfind('\n')
            if last_newline != -1:
                content = content[:last_newline]
            else:
                content = content[:max_chars]

        return {
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": content},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
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
    # Lazily initialized caches
    tokenizer, edu_tokenizer, edu_model = None, None, None

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
    
    def calculate_edu_score(text: str) -> dict:
        """Calculate fineweb edu score; lazily load model on first use."""
        nonlocal edu_tokenizer, edu_model
        if not text or not text.strip():
            return {"score": 0.0, "int_score": 0}
        if edu_tokenizer is None or edu_model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            edu_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
            edu_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
        try:
            inputs = edu_tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
            outputs = edu_model(**inputs)
            logits = outputs.logits.squeeze(-1).float().detach().numpy()
            score = logits.item()
            int_score = int(round(max(0, min(score, 5))))
            return {"score": score, "int_score": int_score}
        except Exception:
            return {"score": 0.0, "int_score": 0}

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
        
        output_text = ""
        if inference_results:
            # Get the first successful result
            for result in inference_results:
                if hasattr(result, 'text'):  # InferenceSuccess object
                    output_text = result.text
                    break
                elif isinstance(result, dict):
                    # Extract from dict format
                    if "choices" in result and len(result["choices"]) > 0:
                        choice = result["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            output_text = choice["message"]["content"]
                            break
                        elif "text" in choice:
                            output_text = choice["text"]
                            break
                    elif "content" in result:
                        output_text = result["content"]
                        break
                    elif "text" in result:
                        output_text = result["text"]
                        break
        
        # Parse thinking and final output
        thinking_text, final_output_text = parse_thinking_output(output_text)
        
        # Calculate metrics for input, thinking, and final output
        input_token_count = count_tokens(document.text)
        thinking_token_count = count_tokens(thinking_text)
        final_output_token_count = count_tokens(final_output_text)
        
        input_edu_scores = calculate_edu_score(document.text)
        thinking_edu_scores = calculate_edu_score(thinking_text)
        final_output_edu_scores = calculate_edu_score(final_output_text)
        
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
            # Get tokens from metadata
            input_tokens = document.metadata["input"]["token_count"]
            thinking_tokens = document.metadata["thinking"]["token_count"]
            final_output_tokens = document.metadata["token_count"]
            
            log_cutoff = 2500
            delimiter_outside = '='*100
            delimiter_inside = '-'*50
            
            # Prepare document structure summary
            def format_document_structure(doc: Document) -> str:
                """Format document structure without showing full content."""
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
            
            document_structure = format_document_structure(document)
            
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
        
        return document
    
    return postprocess_fn


# Create argument parser
parser = argparse.ArgumentParser("Rephrase documents using inference pipeline.")

parser.add_argument(
    "--data_paths", type=str, help="Path to the data to rephrase.", required=True
)
parser.add_argument(
    "--name", "-n", type=str, help="Name of the rephrasing experiment", required=True
)
parser.add_argument(
    "--prompt_template", type=str, help="Path to prompt template file (relative to prompts/ directory)", required=True
)
parser.add_argument(
    "--output_path", type=str, help="Path to the base output folder.", default=S3_BASE_PATH
)
parser.add_argument(
    "--model_name_or_path", type=str, help="Model name or path for inference", default="Qwen/Qwen3-0.6B-FP8"
)
parser.add_argument(
    "--limit", type=int, help="Limit the number of documents to rephrase", default=-1
)
parser.add_argument(
    "--n_tasks", type=int, help="Number of parallel tasks", default=1
)
parser.add_argument(
    "--temperature", type=float, help="Temperature for inference", default=0.7
)
parser.add_argument(
    "--top_p", type=float, help="Top-p (nucleus sampling) for inference", default=0.8
)
parser.add_argument(
    "--top_k", type=int, help="Top-k sampling for inference", default=20
)
parser.add_argument(
    "--presence_penalty", type=float, help="Presence penalty for inference", default=1.5
)
parser.add_argument(
    "--enable_thinking", action="store_true", help="Enable thinking in chat template"
)
parser.add_argument(
    "--model_max_context", type=int, help="Maximum context length for the model", default=16384
)
parser.add_argument(
    "--max_concurrent_requests", type=int, help="Maximum concurrent requests", default=500
)
parser.add_argument(
    "--max_concurrent_tasks", type=int, help="Maximum concurrent tasks", default=500
)
parser.add_argument(
    "--metric_interval", type=int, help="Metric logging interval in seconds", default=60
)
parser.add_argument(
    "--records_per_chunk", type=int, help="Number of records per chunk", default=1000
)
parser.add_argument(
    "--max_tokens", type=int, help="Maximum tokens per request", default=8192 # Should be half of the model context length
)
parser.add_argument(
    "--server_type", type=str, help="Inference server type", choices=["vllm", "sglang", "dummy"], default="vllm"
)
parser.add_argument(
    "--text_key", type=str, default="text", help="Key name for text content in input documents"
)
parser.add_argument(
    "--disable_checkpoints", action="store_true", help="Disable checkpoint functionality"
)
parser.add_argument(
    "--debug", action="store_true", help="Enable debug logging to show input/output text pairs in terminal"
)
parser.add_argument(
    "--tokenizer", type=str, default="hynky/Llama-3.2-1B-no-bos", help="Tokenizer to use for token counting in debug output"
)
parser.add_argument(
    "--time", type=str, default="20:00:00", help="Slurm time limit"
)
parser.add_argument(
    "--partition", type=str, default="hopper-cpu", help="Slurm partition"
)
parser.add_argument(
    "--qos", type=str, default="normal", help="Slurm QoS"
)
parser.add_argument(
    "--dep_job_id", type=str, default=None, help="Optional Slurm dependency job id"
)
parser.add_argument(
    "--gpus", type=int, default=1, help="Number of GPUs per task"
)
parser.add_argument(
    "--enable_prefix_caching", action="store_true", default=True, help="Enable prefix caching"
)
parser.add_argument(
    "--enable_chunked_prefill", action="store_true", default=True, help="Enable chunked prefill"
)
parser.add_argument(
    # https://docs.vllm.ai/en/latest/configuration/optimization.html#performance-tuning-with-chunked-prefill
    "--max_num_batched_tokens", type=int, default=256, help="Maximum number of tokens to batch"
)
parser.add_argument(
    "--max_num_seqs", type=int, default=256, help="Maximum number of sequences to batch"
)
parser.add_argument(
    "--run_local", action="store_true", help="Run pipeline locally instead of using Slurm"
)


def main():
    args = parser.parse_args()
    
    # Set up paths based on arguments
    output_path = f"{args.output_path}/rephrased/{args.name}"
    logs_path = f"{LOG_BASE_PATH}/rephrasing/{args.name}"
    checkpoints_path = f"{LOCAL_TMP_PATH_ON_NODE}/checkpoints/{args.name}" if not args.disable_checkpoints else None

    # Parse data paths
    data_paths = args.data_paths.split(",")
    print(f"Data paths: {data_paths}")
    print(f"Output path: {output_path}")
    
    if args.debug:
        print("🔍 DEBUG MODE ENABLED: Input/output pairs will be logged to terminal")

    config: InferenceConfig = InferenceConfig(
        server_type=args.server_type,
        model_name_or_path=args.model_name_or_path,
        temperature=args.temperature,
        model_max_context=args.model_max_context,
        max_concurrent_requests=args.max_concurrent_requests,
        max_concurrent_tasks=args.max_concurrent_tasks,
        metric_interval=args.metric_interval,
        dp=args.gpus,
        model_kwargs={
            "enable_prefix_caching": args.enable_prefix_caching,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
        },
    )

    # Load prompt template if specified
    try:
        prompt_template = load_prompt_template(args.prompt_template)
        print(f"Loaded prompt template: {args.prompt_template}")
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
            *[JsonlReader(data_path, text_key=args.text_key, limit=args.limit / args.n_tasks) for data_path in data_paths],
            InferenceRunner(
                query_builder=create_templated_query_builder(
                    prompt_template, 
                    args.max_tokens, 
                    args.temperature, 
                    args.top_p, 
                    args.top_k, 
                    args.presence_penalty, 
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
        rephrase_executor = LocalPipelineExecutor(pipeline=pipeline)
    else:
        rephrase_executor = SlurmPipelineExecutor(
            job_name=f"rephrase-{args.name}",
            pipeline=pipeline,
            logging_dir=logs_path,
            tasks=args.n_tasks,
            time=args.time,
            partition="hopper-prod",
            cpus_per_task=10*args.gpus,
            mem_per_cpu_gb=8,
            qos=args.qos,
            env_command="sleep $((RANDOM % 30))",
            mail_user="joel@hf.co",
            depends_job_id=args.dep_job_id,
            sbatch_args={"gres": f"gpu:{args.gpus}"}
        )
    
    rephrase_executor.run()
    

if __name__ == "__main__":
    main()
