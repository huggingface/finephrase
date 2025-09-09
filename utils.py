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


def human_readable(num):
    if num >= 1_000_000_000_000:
        return f"~{round(num / 1_000_000_000_000)}T"
    elif num >= 1_000_000_000:
        return f"~{round(num / 1_000_000_000)}B"
    elif num >= 1_000_000:
        return f"~{round(num / 1_000_000)}M"
    elif num >= 1_000:
        return f"~{round(num / 1_000)}K"
    else:
        return f"{num}"