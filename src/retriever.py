"""Search the persisted BM25 index."""

import json
import os
from typing import List

import bm25s

from src.indexer import tokenize
from src.models import MinimalSource


class Retriever:
    """Load the index once, then answer many queries."""

    def __init__(self, processed_dir: str = "data/processed") -> None:
        """Load the BM25 index and the chunk spans.

        Args:
            processed_dir: Directory written by the index command.

        Raises:
            FileNotFoundError: If the index has not been built.
        """
        bm25_dir = os.path.join(processed_dir, "bm25")
        chunks_path = os.path.join(processed_dir, "chunks.json")
        if not os.path.isdir(bm25_dir) or not os.path.isfile(chunks_path):
            raise FileNotFoundError(
                f"No index in {processed_dir}: run the index command first")
        self.bm25 = bm25s.BM25.load(bm25_dir)
        with open(chunks_path) as f:
            self.chunks = [MinimalSource(**c) for c in json.load(f)]

    def search(self, queries: List[str], k: int) -> List[List[MinimalSource]]:
        """Return the top-k sources for each query, best first.

        Args:
            queries: Questions to search for.
            k: Number of sources per question.

        Returns:
            One list of sources per query; empty when no word is known.
        """
        vocab = self.bm25.vocab_dict
        # Unknown words carry no BM25 score, and a query with none left
        # (empty or nonsense) simply gets no result.
        tokens = [[t for t in q if t in vocab] for q in tokenize(queries)]
        results: List[List[MinimalSource]] = [[] for _ in queries]
        todo = [i for i, t in enumerate(tokens) if t]
        k = min(k, len(self.chunks))
        if not todo or k <= 0:
            return results
        ids, scores = self.bm25.retrieve([tokens[i] for i in todo], k=k,
                                         show_progress=False)
        for row, i in enumerate(todo):
            results[i] = [self.chunks[j] for j, s in
                          zip(ids[row], scores[row]) if s > 0]
        return results
