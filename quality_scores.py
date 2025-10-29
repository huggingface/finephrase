import logging
import threading
import urllib.request
from pathlib import Path
from datatrove.pipeline.base import PipelineStep

from utils import CACHE_PATH

# -----------------------------
# Fineweb EDU score utilities (lazy HF load, like DCLM)
# -----------------------------

_EDU_MODEL_NAME = "HuggingFaceFW/fineweb-edu-classifier"

_EDU_TOKENIZER = None
_EDU_MODEL = None
_EDU_MODEL_LOCK = threading.Lock()
_EDU_MODEL_DOWNLOADED = False

def _download_edu_model_if_needed():
    global _EDU_MODEL_DOWNLOADED
    if _EDU_MODEL_DOWNLOADED:
        return
    # Cache the edu-classifier, since they are loaded with local_files_only=True afterwards
    try:
        logging.info(f"  - Caching {_EDU_MODEL_NAME} (model + tokenizer)...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=_EDU_MODEL_NAME, ignore_patterns=["*.gguf", "*.msgpack", "src/**", "utils/**"])
        _EDU_MODEL_DOWNLOADED = True
    except Exception as e:
        logging.warning(f"Failed to pre-cache EDU classifier: {e}", exc_info=True)

def _load_edu_model_and_tokenizer():
    _download_edu_model_if_needed() # Download model if not already cached
    
    try:
        # Imported here to avoid heavy import at module load
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        logging.warning(f"Failed to import transformers for EDU classifier: {e}")
        raise

    logging.info(f"Loading EDU classifier: {_EDU_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(_EDU_MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(_EDU_MODEL_NAME, local_files_only=True).eval()
    logging.info("EDU classifier loaded.")
    return tokenizer, model


def _get_edu_tokenizer_and_model():
    global _EDU_TOKENIZER, _EDU_MODEL
    if _EDU_MODEL is None or _EDU_TOKENIZER is None:
        with _EDU_MODEL_LOCK:
            if _EDU_MODEL is None or _EDU_TOKENIZER is None:
                _EDU_TOKENIZER, _EDU_MODEL = _load_edu_model_and_tokenizer()
    return _EDU_TOKENIZER, _EDU_MODEL


def calculate_edu_score(text: str) -> float:
    """Return the fineweb-edu score in [0, 5] for given text."""
    if not text or not text.strip():
        return 0.0
    try:
        import torch # Imported here to avoid heavy import at module load
        tokenizer, model = _get_edu_tokenizer_and_model()
        inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        score = float(outputs.logits.squeeze(-1).float().detach().numpy().item())
        return max(0.0, min(score, 5.0))
    except Exception as e:
        logging.warning(f"calculate_edu_score failed; returning 0.0. Error: {e}", exc_info=True)
        return 0.0


# -----------------------------
# DCLM classifier utilities (FastText OH/ELI5)
# -----------------------------

# Model hosted on Hugging Face; we store it under CACHE_PATH for reuse
_DCLM_MODEL_URL = (
    "https://huggingface.co/mlfoundations/fasttext-oh-eli5/resolve/main/"
    "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
)
_DCLM_MODEL_PATH = Path(CACHE_PATH) / "dclm" / "fasttext_oh_eli5.bin"

_DCLM_MODEL = None
_DCLM_MODEL_LOCK = threading.Lock()


def _download_dclm_model_if_needed() -> Path:
    _DCLM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _DCLM_MODEL_PATH.exists():
        logging.info(f"Downloading DCLM classifier model to {_DCLM_MODEL_PATH} ...")
        urllib.request.urlretrieve(_DCLM_MODEL_URL, _DCLM_MODEL_PATH.as_posix())
        logging.info("DCLM classifier model download complete.")
    else:
        logging.info(f"DCLM classifier model already present at {_DCLM_MODEL_PATH}.")


def _load_dclm_model():
    try:
        import fasttext  # Imported here to avoid heavy import at module load
    except Exception as e:
        logging.warning(f"Failed to import fasttext for DCLM classifier: {e}")
        raise

    _download_dclm_model_if_needed()
    logging.info(f"Loading DCLM classifier model from {_DCLM_MODEL_PATH} ...")
    model = fasttext.load_model(_DCLM_MODEL_PATH.as_posix())
    logging.info("DCLM classifier model loaded.")
    return model


def _get_dclm_model():
    global _DCLM_MODEL
    if _DCLM_MODEL is None:
        with _DCLM_MODEL_LOCK:
            if _DCLM_MODEL is None:
                _DCLM_MODEL = _load_dclm_model()
    return _DCLM_MODEL


def calculate_dclm_score(text: str) -> float:
    """Return the DCLM classifier score (probability of __label__hq) for given text.

    Find more information about the classifier here: https://github.com/mlfoundations/dclm/tree/main/baselines#fasttext-filtering

    Returns a float in [0, 1]. On error or empty text, returns 0.0.
    """
    if not text or not text.strip():
        return 0.0
    try:
        threshold = 0.018112 # default
        # FastText expects a single line of text; remove newlines and collapse whitespace
        sanitized_text = " ".join(text.split())
        if not sanitized_text:
            return 0.0
        model = _get_dclm_model()
        labels, probs = model.predict(sanitized_text, k=2, threshold=threshold)
        label_to_prob = {label: prob for label, prob in zip(labels, probs)}
        return float(label_to_prob.get("__label__hq", 0.0))
    except Exception as e:
        logging.warning(f"calculate_dclm_score failed; returning 0.0. Error: {e}", exc_info=True)
        return 0.0




class QualityScoreStatsLogger(PipelineStep):
    """
    Pipeline step that logs quality score statistics (EDU and DCLM).
    """

    type = "📊 - STATS"
    name = "Quality Score Stats Logger"

    def __init__(self):
        super().__init__()

    def run(self, data, rank: int = 0, world_size: int = 1):
        """
        Log education score statistics and pass documents through unchanged.

        Args:
            data: Input documents pipeline
            rank: Worker rank
            world_size: Total number of workers

        Yields:
            Documents unchanged after logging stats
        """
        for doc in data:
            with self.track_time():
                in_meta = doc.metadata.get("input", {})
                thinking_meta = doc.metadata.get("thinking", {})
                out_meta = doc.metadata

                # Token count statistics
                self.stat_update("input_token_count", value=in_meta.get("token_count", 0))
                self.stat_update("thinking_token_count", value=thinking_meta.get("token_count", 0))
                self.stat_update("output_token_count", value=out_meta.get("token_count", 0))
                self.stat_update("token_reduction", value=out_meta.get("token_reduction", 0))

                # EDU score statistics
                self.stat_update("input_edu_score", value=in_meta.get("edu_score", 0.0))
                self.stat_update("thinking_edu_score", value=thinking_meta.get("edu_score", 0.0))
                self.stat_update("output_edu_score", value=out_meta.get("edu_score", 0.0))
                self.stat_update("edu_score_difference", value=out_meta.get("edu_score_difference", 0.0))
                self.stat_update("edu_score_improvement", value=out_meta.get("edu_score_improvement", 0))

                # DCLM score statistics
                self.stat_update("input_dclm_score", value=in_meta.get("dclm_score", 0.0))
                self.stat_update("thinking_dclm_score", value=thinking_meta.get("dclm_score", 0.0))
                self.stat_update("output_dclm_score", value=out_meta.get("dclm_score", 0.0))
                self.stat_update("dclm_score_difference", value=out_meta.get("dclm_score_difference", 0.0))
                self.stat_update("dclm_score_improvement", value=out_meta.get("dclm_score_improvement", 0))

            yield doc
