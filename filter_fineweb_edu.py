import argparse

from utils import LOG_BASE_PATH, S3_BASE_PATH, build_reader


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


parser = argparse.ArgumentParser("Filter fineweb-edu dataset by quality and count tokens.")
parser.add_argument(
    "--data-paths", type=str, help="Path to the data to filter.", required=True
)
parser.add_argument(
    "--output-path", type=str, help="Path to the base output folder. The final output path will be <output_path>/filtered/<name>", default=S3_BASE_PATH
)
parser.add_argument(
    "--quality", type=str, choices=["lq", "hq"], required=True, help="Quality subset to select based on FineWeb-Edu score (lq: <=2, hq: >4)"
)
parser.add_argument(
    "--name", type=str, default=None, help="Name of the filtering. If not provided, the name will be the last part of the data paths"
)
parser.add_argument(
    "--subset-tokens", type=float, help="Number of tokens to subset", default=21.5e9, required=False
)
parser.add_argument(
    "--total-tokens", type=int, help="Total number of tokens. If not set, just counts tokens.", default=None
)
parser.add_argument(
    "--limit", type=int, help="Limit the number of documents to process", default=-1
)
parser.add_argument(
    "--n-tasks", type=int, help="Number of tasks", default=1000
)
parser.add_argument(
    # For avg 100k tokens we can set batch size to 2k for 8cpus with 2gb per cpu
    "--batch-size", type=int, help="Batch size", default=2000
)
parser.add_argument(
    "--qos", type=str, default="normal", help="QoS to use for the job"
)
parser.add_argument(
    "--dep-job-id", type=str, default=None, help="Optional Slurm dependency job id"
)
parser.add_argument(
    "--tokenizer", type=str, default="hynky/Llama-3.2-1B-no-bos", help="Tokenizer to use for counting tokens"
)
parser.add_argument(
    "--sample-seed", type=int, default=42, help="Seed for the sample filter random number generator"
)

def main():
    args = parser.parse_args()
    print(f"Output name: {args.name}")

    data_paths = args.data_paths.split(",")
    print(f"Data paths: {data_paths}")

    _score_predicate = score_predicate_lq if args.quality == "lq" else score_predicate_hq
    
    reader = [build_reader(data_path, limit=args.limit, n_tasks=args.n_tasks, shuffle_files=True) for data_path in data_paths]

    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.tokens import TokensCounter
    
    if args.total_tokens is None:
        # If total tokens is not set, we just count the tokens
        count_executor = SlurmPipelineExecutor(
            job_name=f"count-{args.name}",
            pipeline=[
                *(reader),
                LambdaFilter(filter_function=_score_predicate),
                TokensCounter(tokenizer_name_or_path=args.tokenizer, batch_size=args.batch_size),
            ],
            tasks=args.n_tasks,
            time="1:00:00",
            partition="hopper-cpu",
            logging_dir=f"{LOG_BASE_PATH}/counting/{args.name}",
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
        output_path = f"{args.output_path}/filtered/{args.name}"
        
        filter_executor = SlurmPipelineExecutor(
            job_name=f"filter-{args.name}",
            pipeline=[
                *(reader),
                LambdaFilter(filter_function=_score_predicate),
                SamplerFilter(rate=args.subset_tokens / args.total_tokens, seed=args.sample_seed),
                JsonlWriter(output_path),
            ],
            tasks=args.n_tasks,
            time="5:00:00",
            partition="hopper-cpu",
            logging_dir=f"{LOG_BASE_PATH}/filtering/{args.name}",
            cpus_per_task=8,
            mem_per_cpu_gb=2,
            qos=args.qos,
            env_command="sleep $((RANDOM % 30))",
            mail_user="joel@hf.co",
            depends_job_id=args.dep_job_id
        )
        filter_executor.run()

if __name__ == "__main__":
    main()