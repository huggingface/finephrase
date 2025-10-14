import argparse
from typing import Iterable

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep

from utils import build_reader, LOG_BASE_PATH, print_debug_output


class FormatterStep(PipelineStep):
    name = "Inspect Data Formatter"
    type = "🖨️ - PRINT"

    def __init__(self, limit: int = 10, text_key: str = "text"):
        super().__init__()
        self.limit = limit
        self.text_key = text_key

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        count = 0
        for doc in data:
            # Ensure doc.text is populated if dataset used a custom text key
            if self.text_key != "text" and self.text_key in doc.metadata:
                doc.text = doc.metadata.get(self.text_key, doc.text)

            # For raw inputs without inference results, show text as final_output_text
            print_debug_output(document=doc, thinking_text="", final_output_text=getattr(doc, "text", ""))
            count += 1
            if 0 < self.limit <= count:
                break
        # passthrough
        return data


def main():
    parser = argparse.ArgumentParser("Inspect and pretty-print dataset samples")
    parser.add_argument("--data", type=str, required=True, help="Dataset path (s3:// or hf://)")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to print")
    parser.add_argument("--text-key", type=str, default="text", help="Text field in JSONL records")
    args = parser.parse_args()

    logs_path = f"{LOG_BASE_PATH}/inspections"

    # Build a single-reader pipeline with a local executor
    reader = build_reader(args.data, limit=args.limit, n_tasks=1, shuffle_files=True, text_key=args.text_key)
    pipeline = [reader, FormatterStep(limit=args.limit, text_key=args.text_key)]

    executor = LocalPipelineExecutor(pipeline=pipeline, logging_dir=logs_path, skip_completed=False)
    executor.run()


if __name__ == "__main__":
    main()


