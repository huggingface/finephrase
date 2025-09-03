"""
Chunked inference pipeline example.

This example shows how to run inference on documents using the InferenceRunner
with chunking enabled. Documents are processed in chunks with checkpoint support
for resuming from failures. Each chunk is saved to a separate output file.
"""

import argparse
import os
import sys
from typing import Any

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner


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


def create_templated_query_builder(max_tokens: int, prompt_template: str = None):
    """
    Create a query builder function that uses a prompt template.
    
    Args:
        max_tokens:         Maximum tokens for the response
        prompt_template:    The prompt template content with placeholders
        
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
        }
    return query_builder


def create_logging_postprocess_fn(debug: bool = False):
    """
    Create a postprocess function that optionally logs both input and output text.
    
    Args:
        debug: Whether to enable debug logging
    
    Returns:
        Postprocess function for logging input/output pairs (if debug=True)
    """
    def postprocess_fn(document: Document) -> Document:
        """
        Postprocess function that conditionally logs the input text and paraphrased output.
        
        Args:
            document: Document with inference results in metadata
            
        Returns:
            The document (unchanged)
        """
        if debug:
            # First, log that postprocess function was called
            sys.stdout.write("🔧 DEBUG: Postprocess function called!\n")
            sys.stdout.flush()
            
            # Extract inference results from document metadata
            inference_results = document.metadata.get("inference_results", [])
            sys.stdout.write(f"🔧 DEBUG: Found {len(inference_results)} inference results\n")
            sys.stdout.flush()
            
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
                
                # If still no output, log the structure for debugging
                if not output_text:
                    sys.stdout.write(f"🔧 DEBUG: Result structure: {str(inference_results[0])[:500]}...\n")
                    sys.stdout.flush()
            
            sys.stdout.write(f"🔧 DEBUG: Extracted output length: {len(output_text)}\n")
            sys.stdout.flush()

            log_cutoff = 2500
            
            # Log the input and output pair
            debug_output = f"""
{'='*80}
🔍 DEBUG: INPUT/OUTPUT PAIR
{'='*80}
📝 INPUT TEXT:
{'-'*40}
{document.text[:log_cutoff] + ("..." if len(document.text) > log_cutoff else "")}
{'-'*40}
🔄 PARAPHRASED TEXT:
{'-'*40}
{output_text[:log_cutoff] + ("..." if len(output_text) > log_cutoff else "")}
{'='*80}

"""
            # Write to stdout and flush to ensure immediate visibility
            sys.stdout.write(debug_output)
            sys.stdout.flush()
        
        return document
    
    return postprocess_fn



def simple_query_builder(max_tokens: int):
    """
    Create a query builder function for rephrasing documents.

    Args:
        max_tokens:     Maximum tokens for the response

    Returns:
        Query builder function
    """
    def query_builder(runner: InferenceRunner, document: Document) -> dict[str, Any]:
        """
        Simple query builder that extracts text from document for rephrasing.

        Args:
            runner:     Inference runner instance
            document:   Input document with text content

        Returns:
            Query payload for the inference server
        """
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": document.text},
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }
    return query_builder


# Configuration
USER = os.environ.get('USER')
PROJECT_NAME = "finephrase"

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
    "--output_path", type=str, help="Path to the base output folder.", default=f"s3://{PROJECT_NAME}/experiments/rephrased"
)
parser.add_argument(
    "--model_name_or_path", type=str, help="Model name or path for inference", default="Qwen/Qwen3-4B-FP8"
)
parser.add_argument(
    "--limit", type=int, help="Limit the number of documents to rephrase", default=-1
)
parser.add_argument(
    "--n_tasks", type=int, help="Number of parallel tasks", default=1
)
parser.add_argument(
    "--temperature", type=float, help="Temperature for inference", default=0.5
)
parser.add_argument(
    "--model_max_context", type=int, help="Maximum context length for the model", default=8192
)
parser.add_argument(
    "--max_concurrent_requests", type=int, help="Maximum concurrent requests", default=500
)
parser.add_argument(
    "--max_concurrent_tasks", type=int, help="Maximum concurrent tasks", default=500
)
parser.add_argument(
    "--metric_interval", type=int, help="Metric logging interval in seconds", default=120
)
parser.add_argument(
    "--records_per_chunk", type=int, help="Number of records per chunk", default=500
)
parser.add_argument(
    "--max_tokens", type=int, help="Maximum tokens per request", default=4096
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


def main():
    args = parser.parse_args()
    
    # Set up paths based on arguments
    BASE_PATH = f"/fsx/{USER}"
    output_path = f"{args.output_path}/{args.name}"
    logs_path = f"{BASE_PATH}/logs/{PROJECT_NAME}/experiments/rephrasing/{args.name}"
    checkpoints_path = f"{BASE_PATH}/checkpoints/{args.name}" if not args.disable_checkpoints else None

    # Parse data paths
    data_paths = args.data_paths.split(",")
    print(f"Data paths: {data_paths}")
    print(f"Output path: {output_path}")
    
    # Debug mode confirmation
    if args.debug:
        debug_msg = "🔍 DEBUG MODE ENABLED: Input/output pairs will be logged to terminal\n"
        sys.stdout.write(debug_msg)
        sys.stdout.flush()

    # Import required modules for pipeline
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.executor.local import LocalPipelineExecutor

    # Configure the inference settings with chunking
    # Inspired by Nemotron-CC config (https://arxiv.org/pdf/2412.02595 Section 2.3)
    config: InferenceConfig = InferenceConfig(
        server_type=args.server_type,
        model_name_or_path=args.model_name_or_path,
        temperature=args.temperature,
        model_max_context=args.model_max_context,
        max_concurrent_requests=args.max_concurrent_requests,
        max_concurrent_tasks=args.max_concurrent_tasks,
        metric_interval=args.metric_interval,
    )

    # Load prompt template if specified
    try:
        prompt_template = load_prompt_template(args.prompt_template)
        print(f"Loaded prompt template: {args.prompt_template}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Available templates:")
        prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        for root, dirs, files in os.walk(prompts_dir):
            for file in files:
                if file.endswith('.md'):
                    rel_path = os.path.relpath(os.path.join(root, file), prompts_dir)
                    print(f"  - {rel_path}")
        exit(1)

    # Create the pipeline executor with pipeline defined directly
    pipeline_executor: LocalPipelineExecutor = LocalPipelineExecutor(
        pipeline=[
            *[JsonlReader(data_path, text_key=args.text_key, limit=args.limit) for data_path in data_paths],
            InferenceRunner(
                query_builder=create_templated_query_builder(args.max_tokens, prompt_template),
                config=config,
                records_per_chunk=args.records_per_chunk,
                checkpoints_local_dir=checkpoints_path,
                output_writer=JsonlWriter(output_path, output_filename="${rank}_chunk_${chunk_index}.jsonl"),
                postprocess_fn=create_logging_postprocess_fn(debug=args.debug),
            ),
        ],
        logging_dir=logs_path,
        tasks=args.n_tasks,
    )
    
    # Run the pipeline
    pipeline_executor.run()

if __name__ == "__main__":
    main()
