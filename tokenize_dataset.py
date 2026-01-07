import argparse

from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep

from utils import ENV_COMMAND, LOCAL_TMP_PATH_ON_NODE, LOG_BASE_PATH, S3_BASE_PATH, build_reader


class DocumentSplitter(PipelineStep):
    def __init__(self, max_chars_per_document: int):
        super().__init__()
        self.max_chars_per_document = max_chars_per_document
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        from datatrove.data import Document
        for document in data:
            if len(document.text) <= self.max_chars_per_document:
                yield document
            else:
                # Split the document into smaller chunks
                chunks = self._split_text(document.text, self.max_chars_per_document)
                for i, chunk in enumerate(chunks):
                    # Create new document for each chunk
                    new_doc = Document(
                        text=chunk,
                        id=f"{document.id}_chunk_{i}" if document.id else f"chunk_{i}",
                        metadata=document.metadata.copy() if document.metadata else {}
                    )
                    yield new_doc
    
    def _split_text(self, text: str, max_chars: int) -> list[str]:
        """Split text into chunks, preferring to split on \\n\\n when possible."""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        remaining_text = text
        
        while len(remaining_text) > max_chars:
            # Find the best split point within the character limit
            split_point = self._find_best_split_point(remaining_text, max_chars)
            
            # Extract the chunk and update remaining text
            chunk = remaining_text[:split_point].rstrip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move to the next part, skipping any leading whitespace
            remaining_text = remaining_text[split_point:].lstrip()
        
        # Add the final chunk if it's not empty
        if remaining_text.strip():
            chunks.append(remaining_text.strip())
        
        return chunks
    
    def _find_best_split_point(self, text: str, max_chars: int) -> int:
        """Find the best point to split text within max_chars limit."""
        if len(text) <= max_chars:
            return len(text)
        
        # Try splitting on double newlines first
        double_newline_splits = []
        pos = 0
        while pos < max_chars:
            pos = text.find('\n\n', pos)
            if pos == -1 or pos >= max_chars:
                break
            double_newline_splits.append(pos + 2)  # Include the \n\n
            pos += 2
        
        if double_newline_splits:
            return double_newline_splits[-1]
        
        # Try splitting on single newlines
        single_newline_splits = []
        pos = 0
        while pos < max_chars:
            pos = text.find('\n', pos)
            if pos == -1 or pos >= max_chars:
                break
            single_newline_splits.append(pos + 1)  # Include the \n
            pos += 1
        
        if single_newline_splits:
            return single_newline_splits[-1]
        
        # Try splitting on sentence boundaries
        sentence_splits = []
        pos = 0
        while pos < max_chars:
            pos = text.find('. ', pos)
            if pos == -1 or pos >= max_chars:
                break
            sentence_splits.append(pos + 2)  # Include the '. '
            pos += 2
        
        if sentence_splits:
            return sentence_splits[-1]
        
        # Last resort: split at word boundaries
        last_space = text.rfind(' ', 0, max_chars)
        if last_space > 0:
            return last_space + 1
        
        # Absolute last resort: hard split at character limit
        return max_chars


class PruneEmptyTokenizedFiles(PipelineStep):
    def __init__(self, input_folder: str):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)

    def run(self, data, rank: int = 0, world_size: int = 1):
        assert world_size == 1, "world_size must be 1 for pruning empty tokenized files"

        datafiles = self.input_folder.list_files(glob_pattern="*.ds")
        for datafile in datafiles:
            index_path = f"{datafile}.index"
            if not self.input_folder.exists(index_path):
                continue

            with self.input_folder.open(index_path, "rb") as index_file:
                if index_file.read(1):
                    continue

            self.input_folder.rm_file(datafile)
            self.input_folder.rm_file(index_path)

            metadata_path = f"{datafile}.metadata"
            if self.input_folder.exists(metadata_path):
                self.input_folder.rm_file(metadata_path)

            loss_path = f"{datafile}.loss"
            if self.input_folder.exists(loss_path):
                self.input_folder.rm_file(loss_path)

        return data

parser = argparse.ArgumentParser("Sample and tokenize a dataset.")

parser.add_argument(
    "--data", type=str, required=True, help="Path to the data to tokenize. Comma separated list of paths for mixed datasets."
)
parser.add_argument(
    "--name", type=str, required=True, help="Name of the tokenization."  
)
parser.add_argument(
    "--output-path", type=str, help="Path to the base output folder. The final output path will be <output_path>/tokenized/<name>", default=S3_BASE_PATH
)
parser.add_argument(
    "--limit", type=int, help="limit the number of documents to tokenize", default=-1
)
parser.add_argument(
    "--n-tasks", type=int, help="number of tokenization tasks", default=100
)
parser.add_argument(
    "--max-toks", type=int, help="max tokens per file", default=1e8
)
parser.add_argument(
    # For avg 100k tokens we can set batch size to 2k for 8cpus with 2gb per cpu
    "--batch-size", type=int, help="batch size", default=2000
)
parser.add_argument(
    "--qos", type=str, default="normal"
)
parser.add_argument(
    "--tokenizer", type=str, help="tokenizer to use", default="hynky/Llama-3.2-1B-no-bos"
)
parser.add_argument(
    "--text-key", type=str, default="text", help="Use comma-separated keys to concatenate per row, e.g. 'title,body'."
)
parser.add_argument(
    "--sample", type=float, default=1.0
)
parser.add_argument(
    "--dep-job-id", type=str, default=None, help="ID of the job that produced the data to tokenize."
)
parser.add_argument(
    "--jsonl-output", type=str, default=None, help="Path to optionally save the sampled data jsonl"
)
parser.add_argument(
    "--shuffle-chunk-size", type=int, default=4096, help="Shuffle inter document"
)
parser.add_argument(
    "--run-merger", type=bool, default=True, help="Run the merger after tokenization"
)
parser.add_argument(
    "--max-chars-per-document", type=int, default=100_000, help="Split documents larger than this many characters"
)
parser.add_argument(
    "--duplicate", type=int, default=1, help="Duplicate the data n times"
)
parser.add_argument(
    "--sample-seed", type=int, default=42, help="Seed for the sample filter random number generator"
)
parser.add_argument(
    "--shuffle-seed", type=int, default=42, help="Seed for the tokenizer shuffle random number generator"
)
parser.add_argument(
    "--file-type", type=str, default=None, choices=["jsonl", "parquet"], help="Explicit file type (auto-detect if not set)"
)
parser.add_argument(
    "--compression", type=str, default=None, help="Explicit compression (e.g., 'zstd', 'gzip'). Auto-detects if not set."
)


def main():
    args = parser.parse_args()
    print(f"Output name: {args.name}")

    data = args.data.split(",")
    print(f"Data paths: {data}")

    logging_base_path = f"{LOG_BASE_PATH}/tokenization/{args.name}"

    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.filters import SamplerFilter
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
    from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
    
    tokenizer_executor = SlurmPipelineExecutor(
        job_name=f"tok-{args.name}",
        pipeline=[
            *([build_reader(data_path, limit=args.limit, n_tasks=args.n_tasks, shuffle_files=True, text_key=args.text_key, file_type=args.file_type, compression=args.compression) for data_path in data]),
            SamplerFilter(rate=args.sample, seed=args.sample_seed),
            *([JsonlWriter(args.jsonl_output)] if args.jsonl_output else []),
            *([DocumentSplitter(args.max_chars_per_document)] if args.max_chars_per_document else []),
            DocumentTokenizer(
                output_folder=f"{args.output_path}/tokenized/{args.name}",
                local_working_dir=f"{LOCAL_TMP_PATH_ON_NODE}/tokenized/{args.name}",
                eos_token="<|end_of_text|>",
                tokenizer_name_or_path=args.tokenizer,
                batch_size=args.batch_size,
                max_tokens_per_file=args.max_toks,
                # Max 1 GT per file (i.e. btw 5 et 300 tokenized files per dump et about 100 dump extracts per merged file)
                shuffle_documents=True,
                shuffle_chunk_size=args.shuffle_chunk_size + 1 if args.shuffle_chunk_size else None,
                seed=args.shuffle_seed,
            ),
        ],
        tasks=args.n_tasks,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir=logging_base_path,
        cpus_per_task=8,
        mem_per_cpu_gb=2,
        qos=args.qos,
        env_command=ENV_COMMAND,
        mail_user="joel@hf.co",
        depends_job_id=args.dep_job_id
    )

    if args.run_merger:
        merge_executor = SlurmPipelineExecutor(
                job_name=f"merge-{args.name}",
                pipeline=[
                PruneEmptyTokenizedFiles(
                    input_folder=f"{args.output_path}/tokenized/{args.name}",
                ),
                DocumentTokenizerMerger(
                    input_folder=f"{args.output_path}/tokenized/{args.name}",
                    output_folder=f"{args.output_path}/tokenized_merged/{args.name}",
                    save_filename="tokenized_dataset",
                    max_tokens_per_file=args.max_toks,
                    shuffle_chunk_size=args.shuffle_chunk_size + 1 if args.shuffle_chunk_size else None
                ),
            ],
            tasks=1,
            time="20:00:00",
            partition="hopper-cpu",
            logging_dir=f"{logging_base_path}/tokenized_merged",
            cpus_per_task=2,
            mem_per_cpu_gb=32,
            qos=args.qos,
            depends=tokenizer_executor
        )

        merge_executor.run()
    else:
        tokenizer_executor.run() 

if __name__ == "__main__":
    main()
