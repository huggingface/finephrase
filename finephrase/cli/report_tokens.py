
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline

import argparse

from finephrase.utils import ENV_COMMAND, LOG_BASE_PATH, MAIL_USER, build_reader, human_readable


class ReportTokens(PipelineStep):
    def report(self, i, all_tokens):
        print(f"Processed {human_readable(i)} documents")
        print(f"Average tokens: {all_tokens/(i+1):.2f}")
        print(f"Total tokens: {all_tokens} ({human_readable(all_tokens)})")


    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        all_tokens = 0
        for i,document in enumerate(data):
            all_tokens += document.metadata["token_count"]
            if i % 10_000 == 0 and i > 0:
                self.report(i, all_tokens)

        self.report(i+1, all_tokens)


parser = argparse.ArgumentParser(description="Report token statistics for a dataset")
parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Comma-separated list of dataset paths (e.g., s3://..., hf://...)",
)
parser.add_argument(
    "--run-local",
    action="store_true",
    help="Run the pipeline locally, usually for debugging",
)
parser.add_argument(
    "--limit",
    type=int,
    default=-1,
    help="Optional limit on number of samples per input path",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="hynky/Llama-3.2-1B-no-bos",
    help="Hugging Face tokenizer name or path",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=10000,
    help="Batch size for token counting",
)
parser.add_argument(
    "--n-tasks",
    type=int,
    default=100,
    help="Number of parallel tasks",
)

def main():
    args = parser.parse_args()

    logs_path = f"{LOG_BASE_PATH}/token_counts/{args.data.replace('/', '_')}"

    data_paths = args.data.split(",")
    reader = [build_reader(data_path, limit=args.limit, n_tasks=args.n_tasks, shuffle_files=True) for data_path in data_paths]

    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.executor.slurm import SlurmPipelineExecutor
    from datatrove.pipeline.tokens import TokensCounter

    pipeline = [
        *(reader),
        TokensCounter(tokenizer_name_or_path=args.tokenizer, batch_size=args.batch_size),
        ReportTokens(),
    ]
    if args.run_local:
        executor = LocalPipelineExecutor(pipeline)
    else:
        executor = SlurmPipelineExecutor(
            pipeline, 
            tasks=args.n_tasks, 
            time="01:00:00", 
            partition="hopper-cpu", 
            qos="normal", 
            cpus_per_task=4,
            env_command=ENV_COMMAND,
            logging_dir=logs_path, job_name=f"report-tokens-{args.data}", mail_user=MAIL_USER,
        )
    executor.run()

if __name__ == "__main__":
    main()
