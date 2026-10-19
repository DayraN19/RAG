"""Command-line interface for the RAG project."""

import fire


class Cli:
    """Commands exposed through Python Fire."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build the index from data/raw/ into data/processed/.

        Args:
            max_chunk_size: Maximum size of a chunk in characters.
        """
        print(f"index called with max_chunk_size={max_chunk_size}")

    def search(self, query: str, k: int = 10) -> None:
        """Print the top-k sources for a single query.

        Args:
            query: The question to search for.
            k: Number of sources to return.
        """
        print(f"search called with query={query!r}, k={k}")


if __name__ == "__main__":
    fire.Fire(Cli)
