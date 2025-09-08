from datatrove.pipeline.readers import JsonlReader, ParquetReader


def get_reader(path):
    if path.startswith("hf://"):
        # hf datasets are usually in Parquet format
        return ParquetReader
    elif path.startswith("s3://"):
        # s3 datasets are usually in JSONL format
        return JsonlReader
    else:
        raise ValueError(f"Invalid path: {path}")