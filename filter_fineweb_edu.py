import argparse
import os
from datatrove.pipeline.base import PipelineStep

USER = os.environ.get('USER')
PROJECT_NAME = "finephrase"


parser = argparse.ArgumentParser("Quickly launch thom's style of tokenization.")

parser.add_argument(
    "--data_paths", type=str, help="Path to the data to filter.", required=True
)
parser.add_argument(
    "--output_path", type=str, help="Path to the base output folder. The final output path will be <output_path>/filtered/<name>", default=f"s3://{PROJECT_NAME}/experiments"
)
parser.add_argument(
    "--quality", type=str, choices=["lq", "hq"], required=True, help="Quality subset to select based on FineWeb-Edu score (lq: <=2, hq: >4)"
)
parser.add_argument(
    "--subset_tokens", type=float, default=36e9, help="Target number of tokens for the subset (used to compute sampling rate externally)"
)
parser.add_argument(
    "--name", "-n", type=str, default=None, help="Name of the tokenization. If not provided, the name will be the last part of the data paths"
)
parser.add_argument(
    "--limit", type=int, help="limit the number of documents to process", default=-1
)
parser.add_argument(
    "--n_tasks", type=int, help="nb of filtering tasks", default=100
)
# For avg 100k tokens we can set batch size to 2k for 8cpus with 2gb per cpu
parser.add_argument(
    "--batch_size", type=int, help="batch size", default=2000
)
parser.add_argument(
    "--qos", type=str, default="normal"
)
parser.add_argument(
    "--dep_job_id", type=str, default=None, help="Optional Slurm dependency job id"
)
parser.add_argument(
    "--tokenizer", type=str, default="hynky/Llama-3.2-1B-no-bos", help="Tokenizer to use for counting tokens"
)

def score_predicate_lq(doc):
    try:
        score = float(doc.metadata["score"])
        return score <= 2
    except (TypeError, ValueError):
        return False


def score_predicate_hq(doc):
    try:
        score = float(doc.metadata["score"])
        return score > 4
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    args = parser.parse_args()
    # Output name should be the same as last part of the data path
    if args.name:
        output_name = args.name
    else:
        output_name = args.data_paths.replace("/", "_")
    print(f"Output name: {output_name}")

    data_paths = args.data_paths.split(",")
    print(f"Data paths: {data_paths}")

    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.tokens import TokensCounter
    
    logging_base_path = f"/fsx/{USER}/logs/{PROJECT_NAME}/experiments/filtering/{output_name}"
    
    _score_predicate = score_predicate_lq if args.quality == "lq" else score_predicate_hq
    
    output_path = f"{args.output_path}/filtered/{output_name}"
    
    filter_executor = SlurmPipelineExecutor(
        job_name=f"filter-{output_name}",
        pipeline=[
            *([JsonlReader(data_path, shuffle_files=True, limit=args.limit) for data_path in data_paths]),
            LambdaFilter(filter_function=_score_predicate),
            TokensCounter(tokenizer_name_or_path=args.tokenizer, batch_size=args.batch_size),
            JsonlWriter(output_path),
        ],
        tasks=args.n_tasks,
        time="1:00:00",
        partition="hopper-cpu",
        logging_dir=f"{logging_base_path}/filtered",
        cpus_per_task=8,
        mem_per_cpu_gb=2,
        qos=args.qos,
        env_command="sleep $((RANDOM % 30))",
        mail_user="joel@hf.co",
        depends_job_id=args.dep_job_id
    )
    filter_executor.run()
