import argparse

from finephrase.utils import ENV_COMMAND, LOG_BASE_PATH, MAIL_USER, S3_BASE_PATH, build_reader


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


# Predefined filter functions to choose from (extend as needed)
FILTER_FUNCTIONS = {
    "noop": lambda doc: True,
    "fineweb_edu_lq": score_predicate_lq,
    "fineweb_edu_hq": score_predicate_hq,
}


parser = argparse.ArgumentParser("Filter datasets with selectable filter functions and optional subsampling.")
parser.add_argument(
    "--data", type=str, help="Path to the data to filter.", required=True
)
parser.add_argument(
    "--output-path", type=str, help="Path to the base output folder. The final output path will be <output_path>/filtered/<name>", default=S3_BASE_PATH
)
parser.add_argument(
    "--filter", type=str, choices=list(FILTER_FUNCTIONS.keys()), required=True, help="Filter function to apply (use 'noop' to copy all data)"
)
parser.add_argument(
    "--name", type=str, default=None, help="Name of the filtering. If not provided, the name will be the last part of the data paths"
)
parser.add_argument(
    "--subset-tokens", type=float, help="Approximate number of tokens to subset (used only if --total-tokens is provided)", default=None, required=False
)
parser.add_argument(
    "--total-tokens", type=int, help="Total number of tokens in the source data (used to compute sample rate). If not set, no subsampling is applied.", default=None
)
parser.add_argument(
    "--limit", type=int, help="Limit the number of documents to process", default=-1
)
parser.add_argument(
    "--n-tasks", type=int, help="Number of tasks", default=1000
)
parser.add_argument(
    "--qos", type=str, default="normal", help="QoS to use for the job"
)
parser.add_argument(
    "--dep-job-id", type=str, default=None, help="Optional Slurm dependency job id"
)
parser.add_argument(
    "--sample-seed", type=int, default=42, help="Seed for the sample filter random number generator"
)

def main():
    args = parser.parse_args()

    data_paths = args.data.split(",")
    print(f"Data paths: {data_paths}")

    # Derive default name from data paths if not provided
    if not args.name:
        base_parts = []
        for p in data_paths:
            p = p.rstrip("/")
            base_parts.append(p.split("/")[-1] if "/" in p else p)
        args.name = "+".join(base_parts)
    print(f"Output name: {args.name}")

    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
    from datatrove.pipeline.writers import JsonlWriter

    # Optional subsampling if both values are provided
    do_subsampling = args.subset_tokens is not None and args.total_tokens is not None and args.total_tokens > 0

    pipeline = [
        *[build_reader(data_path, limit=args.limit, n_tasks=args.n_tasks, shuffle_files=True) for data_path in data_paths],
        *([LambdaFilter(filter_function=FILTER_FUNCTIONS[args.filter])] if args.filter != "noop" else []),
        *([SamplerFilter(rate=args.subset_tokens / args.total_tokens, seed=args.sample_seed)] if do_subsampling else []),
        JsonlWriter(f"{args.output_path}/filtered/{args.name}")
    ]

    filter_executor = SlurmPipelineExecutor(
        job_name=f"filter-{args.name}",
        pipeline=pipeline,
        tasks=args.n_tasks,
        time="5:00:00",
        partition="hopper-cpu",
        logging_dir=f"{LOG_BASE_PATH}/filtering/{args.name}",
        cpus_per_task=8,
        mem_per_cpu_gb=2,
        qos=args.qos,
        env_command=ENV_COMMAND,
        mail_user=MAIL_USER,
        depends_job_id=args.dep_job_id
    )
    filter_executor.run()

if __name__ == "__main__":
    main()