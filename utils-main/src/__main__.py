import time
import fire
from pathlib import Path
from .file_indexer import build_corpus, build_4_bm25, save_index
from .file_searcher import search, search_dataset
from .file_generation_batch import answer, answer_dataset
from .file_evaluate import evaluate
from .utils import valid_positive_int


def index(max_chunk_size: int = 2000) -> None:
    """Discover, chunk, index and persist every file under data/raw/.

    Args:
        max_chunk_size: Maximum number of characters per chunk.
    """

    if not valid_positive_int(max_chunk_size, "max_chunk_size", 2000):
        return None

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    start = time.time()

    chunks = build_corpus(raw_dir, max_chunk_size)
    bm25 = build_4_bm25(chunks)
    save_index(bm25, chunks, processed_dir)

    elapsed = time.time() - start
    print(
        f"Ingestion complete! {len(chunks)} chunks indexed in {elapsed:.2f}s. "
        f"Indices saved under {processed_dir}/"
    )


def main() -> None:
    """Expose the pipeline commands through Python Fire."""

    fire.Fire({"index": index, "search": search,
               "search_dataset": search_dataset,
               "answer": answer,
               "answer_dataset": answer_dataset,
               "evaluate": evaluate})


if __name__ == "__main__":
    main()
