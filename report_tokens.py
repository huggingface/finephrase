import os
from datatrove.data import DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens import TokensCounter
import argparse

from utils import get_reader, human_readable

USER = os.environ.get('USER')
PROJECT_NAME = "finephrase"

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
    "--data_paths",
    type=str,
    required=True,
    help="Comma-separated list of dataset paths (e.g., s3://..., hf://...)",
)
parser.add_argument(
    "--run_local",
    action="store_true",
    help="Run the pipeline locally",
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
    "--batch_size",
    type=int,
    default=10000,
    help="Batch size for token counting",
)

def main():
    args = parser.parse_args()

    BASE_PATH = f"/fsx/{USER}"
    logs_path = f"{BASE_PATH}/logs/{PROJECT_NAME}/experiments/token_counts/{args.data_paths.replace('/', '_')}"
    
    data_paths = args.data_paths.split(",")
    
    reader = [get_reader(data_path)(data_path, shuffle_files=True, limit=args.limit) for data_path in data_paths]

    pipeline = [
        *(reader),
        TokensCounter(tokenizer_name_or_path=args.tokenizer, batch_size=args.batch_size),
        ReportTokens(),
    ]
    if args.run_local:
        executor = LocalPipelineExecutor(pipeline)
    else:
        executor = SlurmPipelineExecutor(
            pipeline, tasks=100, time="01:00:00", partition="hopper-cpu", qos="normal", cpus_per_task=4,
            logging_dir=logs_path, job_name=f"report-tokens-{args.data_paths}", mail_user="joel@hf.co",
        )
    executor.run()

if __name__ == "__main__":
    main()