"""N-gram overlap contamination audit between training data and eval benchmarks.

Provides a single :class:`NGramContaminationAuditor` pipeline step that scans
documents for n-gram matches against a pre-built decontamination index produced
by :class:`datatrove.pipeline.decont.NGramsDecontIndexer`.

Unlike :class:`datatrove.pipeline.decont.NGramsDecontFilter`, this step:

* never drops documents (audit only),
* checks every n-gram against every benchmark task (not just the first match),
* tallies per-task and per-benchmark-group stats so contamination rates can be
  computed after the run.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.decont.n_grams import NGramsDecontConfig
from datatrove.utils.binaryio import read_np_from_file
from datatrove.utils.hashing import create_hash_func
from datatrove.utils.logging import logger
from datatrove.utils.text import ngrams, simplify_text
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer

# Skip degenerate n-grams where a single token dominates the window (e.g.
# "0 0 0 0 ..." that simplify_text can produce when norm_numbers collapses
# digit runs, or repeated-character spam). Lookups for these would generate
# spurious cross-task matches regardless of whether they're in the index.
_DEGENERATE_MAX_TOKEN_FREQ = 0.6


def _benchmark_group(task_name: str) -> str:
    """Strip the ``suite|`` prefix and ``:subset`` suffix to get a benchmark group.

    >>> _benchmark_group("lighteval|mmlu_redux_cf:anatomy")
    'mmlu_redux_cf'
    >>> _benchmark_group("lighteval|gsm8k")
    'gsm8k'
    """
    name = task_name.split("|", 1)[1] if "|" in task_name else task_name
    return name.split(":", 1)[0]


def _is_degenerate_ngram(tokens: list[str]) -> bool:
    """True if a single token covers more than _DEGENERATE_MAX_TOKEN_FREQ of the window."""
    if not tokens:
        return True
    most_common = max(tokens.count(t) for t in set(tokens))
    return most_common / len(tokens) > _DEGENERATE_MAX_TOKEN_FREQ


class NGramContaminationAuditor(PipelineStep):
    """Audit (don't filter) documents for n-gram overlap with eval benchmarks.

    For every document we compute n-grams, hash them, and look up which
    benchmark tasks each hash belongs to. A document is "contaminated by task T"
    if it contains at least one n-gram whose hash is in T's index. Per-task and
    overall contamination counts are accumulated through
    :meth:`PipelineStep.stat_update` so they end up in ``stats.json``.

    Stats emitted (final values are aggregated across all ranks):

    * ``total_docs`` -- number of documents scanned
    * ``contaminated`` -- docs matching at least one task
    * ``contaminated__<task_name>`` -- docs matching this specific lighteval task
      (e.g. ``lighteval|mmlu_redux_cf:anatomy``)
    * ``contaminated_group__<benchmark>`` -- docs matching any subtask in the
      benchmark group (e.g. ``mmlu_redux_cf`` covers all MMLU subsets)
    """

    type = "DECONT"
    name = "N-grams audit"

    def __init__(
        self,
        index_folder: DataFolderLike,
        config: NGramsDecontConfig | None = None,
        language: str = Languages.english,
    ):
        super().__init__()
        self.index_folder = get_datafolder(index_folder)
        self.config = config or NGramsDecontConfig()
        self.language = language
        self.hash_func = create_hash_func(self.config.hash_config)
        self.tokenizer = load_word_tokenizer(language)
        # Lazy-loaded: hash -> set of task names matching that hash.
        self._hash_to_tasks: dict[int, set[str]] | None = None

    def load_index(self) -> None:
        """Load all ``*.index.hashes`` files into a single hash -> {tasks} map."""

        def _load(file: str) -> tuple[str, list[int]]:
            with self.index_folder.open(file, mode="rb") as f:
                arr = read_np_from_file(
                    f, np.dtype(self.config.hash_config.np_descr), self.index_folder.is_local()
                )
            return file, arr.tolist()

        files = self.index_folder.list_files(glob_pattern="**/*.index.hashes")
        with ThreadPoolExecutor() as pool:
            loaded = list(pool.map(_load, files))

        hash_to_tasks: dict[int, set[str]] = defaultdict(set)
        for filename, hashlist in loaded:
            # Filenames are written as "<task_name>.index.hashes" by the indexer.
            taskname = filename.rsplit("/", 1)[-1].removesuffix(".index.hashes")
            logger.info(f"Loaded {len(hashlist)} hashes for {taskname}")
            for h in hashlist:
                hash_to_tasks[h].add(taskname)
        self._hash_to_tasks = hash_to_tasks
        logger.info(
            f"Index ready: {len(self._hash_to_tasks)} unique hashes across {len(files)} tasks"
        )

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if self._hash_to_tasks is None:
            self.load_index()
        index = self._hash_to_tasks

        for doc in data:
            self.stat_update("total_docs")
            tokens = self.tokenizer.word_tokenize(simplify_text(doc.text, self.config.norm_config))
            matched_tasks: set[str] = set()
            # Walk every n-gram; collect every task whose index contains its hash.
            for ngram_tokens in ngrams(tokens, self.config.n_grams):
                if _is_degenerate_ngram(ngram_tokens):
                    continue
                ngram_hash = self.hash_func(" ".join(ngram_tokens))
                if ngram_hash in index:
                    matched_tasks.update(index[ngram_hash])
            if matched_tasks:
                self.stat_update("contaminated")
                matched_groups: set[str] = set()
                for t in matched_tasks:
                    self.stat_update(f"contaminated__{t}")
                    matched_groups.add(_benchmark_group(t))
                for g in matched_groups:
                    self.stat_update(f"contaminated_group__{g}")
            yield doc
