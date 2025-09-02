"""
Chunked inference pipeline example.

This example shows how to run inference on documents using the InferenceRunner
with chunking enabled. Documents are processed in chunks with checkpoint support
for resuming from failures. Each chunk is saved to a separate output file.
"""

import argparse
import os
from typing import Any

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner



def simple_query_builder(max_tokens: int):
    """
    Create a query builder function for rephrasing documents.

    Args:
        max_tokens: Maximum tokens for the response

    Returns:
        Query builder function
    """
    def query_builder(runner: InferenceRunner, document: Document) -> dict[str, Any]:
        """
        Simple query builder that extracts text from document for rephrasing.

        Args:
            runner: Inference runner instance
            document: Input document with text content

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


if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set up paths based on arguments
    BASE_PATH = f"/fsx/{USER}"
    output_path = f"{args.output_path}/{args.name}"
    logs_path = f"{BASE_PATH}/logs/{PROJECT_NAME}/experiments/rephrased/{args.name}"
    checkpoints_path = f"{BASE_PATH}/checkpoints/{args.name}" if not args.disable_checkpoints else None

    # Parse data paths
    data_paths = args.data_paths.split(",")
    print(f"Data paths: {data_paths}")
    print(f"Output path: {output_path}")

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


    # Create the pipeline executor with pipeline defined directly
    pipeline_executor: LocalPipelineExecutor = LocalPipelineExecutor(
        pipeline=[
            *[JsonlReader(data_path, text_key=args.text_key, limit=args.limit) for data_path in data_paths],
            InferenceRunner(
                query_builder=simple_query_builder(args.max_tokens),
                config=config,
                records_per_chunk=args.records_per_chunk,
                checkpoints_local_dir=checkpoints_path,
                output_writer=JsonlWriter(output_path, output_filename="${rank}_chunk_${chunk_index}.jsonl"),
                postprocess_fn=None,
            ),
        ],
        logging_dir=logs_path,
        tasks=args.n_tasks,
    )
    
    # Run the pipeline
    pipeline_executor.run()

