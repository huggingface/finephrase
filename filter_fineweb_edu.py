import argparse
import os

USER = os.environ.get('USER')
PROJECT_NAME = "finephrase"


parser = argparse.ArgumentParser("Filter fineweb-edu dataset by quality and count tokens.")

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
    "--name", "-n", type=str, default=None, help="Name of the filtering. If not provided, the name will be the last part of the data paths"
)
parser.add_argument(
    "--subset_tokens", type=int, help="Number of tokens to subset", default=36e9, required=False
)
parser.add_argument(
    "--total_tokens", type=int, help="Total number of tokens. If not set, just counts tokens.", default=None
)
parser.add_argument(
    "--limit", type=int, help="Limit the number of documents to process", default=-1
)
parser.add_argument(
    "--n_tasks", type=int, help="Number of tasks", default=1000
)
# For avg 100k tokens we can set batch size to 2k for 8cpus with 2gb per cpu
parser.add_argument(
    "--batch_size", type=int, help="Batch size", default=2000
)
parser.add_argument(
    "--qos", type=str, default="normal", help="QoS to use for the job"
)
parser.add_argument(
    "--dep_job_id", type=str, default=None, help="Optional Slurm dependency job id"
)
parser.add_argument(
    "--tokenizer", type=str, default="hynky/Llama-3.2-1B-no-bos", help="Tokenizer to use for counting tokens"
)

def score_predicate_lq(doc):
    try:
        score = float(doc.metadata["int_score"])
        return score in [0, 1]
    except (TypeError, ValueError):
        return False


def score_predicate_hq(doc):
    try:
        score = float(doc.metadata["int_score"])
        return score in [4, 5]
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
    from datatrove.pipeline.filters import SamplerFilter
    from datatrove.pipeline.readers import JsonlReader, ParquetReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.tokens import TokensCounter
    
    _score_predicate = score_predicate_lq if args.quality == "lq" else score_predicate_hq

    if args.quality == "lq": # LQ data is in JSONL format (edu_annotated on S3)
        reader = [JsonlReader(data_path, shuffle_files=True, limit=args.limit) for data_path in data_paths]
    else: # HQ data is in Parquet format (finweb-edu on the hub)
        reader = [ParquetReader(data_path, shuffle_files=True, limit=args.limit) for data_path in data_paths]
    
    if args.total_tokens is None:
        # If total tokens is not set, we just count the tokens
        count_executor = SlurmPipelineExecutor(
            job_name=f"count-{output_name}",
            pipeline=[
                *(reader),
                LambdaFilter(filter_function=_score_predicate),
                TokensCounter(tokenizer_name_or_path=args.tokenizer, batch_size=args.batch_size),
            ],
            tasks=args.n_tasks,
            time="1:00:00",
            partition="hopper-cpu",
            logging_dir=f"/fsx/{USER}/logs/{PROJECT_NAME}/experiments/counting/{output_name}/counted",
            cpus_per_task=8,
            mem_per_cpu_gb=2,
            qos=args.qos,
            env_command="sleep $((RANDOM % 30))",
            mail_user="joel@hf.co",
            depends_job_id=args.dep_job_id
        )
        count_executor.run()
    else:
        # If total tokens is set, we subsample the data and save it
        output_path = f"{args.output_path}/filtered/{output_name}"
        
        filter_executor = SlurmPipelineExecutor(
            job_name=f"filter-{output_name}",
            pipeline=[
                *(reader),
                LambdaFilter(filter_function=_score_predicate),
                SamplerFilter(rate=args.subset_tokens / args.total_tokens),
                JsonlWriter(output_path),
            ],
            tasks=args.n_tasks,
            time="5:00:00",
            partition="hopper-cpu",
            logging_dir=f"/fsx/{USER}/logs/{PROJECT_NAME}/experiments/filtering/{output_name}/filtered",
            cpus_per_task=8,
            mem_per_cpu_gb=2,
            qos=args.qos,
            env_command="sleep $((RANDOM % 30))",
            mail_user="joel@hf.co",
            depends_job_id=args.dep_job_id
        )
        filter_executor.run()
