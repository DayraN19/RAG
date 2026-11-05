import numpy as np
from pathlib import Path
from tqdm import tqdm
import pydantic
from rank_bm25 import BM25Okapi
from .file_indexer import load_index, IndexedChunk, tokenize
from .models import MinimalSource, MinimalSearchResults
from .utils import valid_positive_int, valid_query
from .models import StudentSearchResults, UnansweredQuestion, RagDataset


# documentation chunks get a small score bonus over source code
DOC_EXTENSIONS = {".md", ".rst", ".txt"}
DOC_BOOST = 1.2


def search_one(question: UnansweredQuestion, k: int,
               bm25: BM25Okapi,
               chunks: list[IndexedChunk]) -> MinimalSearchResults:
    """
    Perform a BM25-based search on a collection of text chunks and return
        the top-k results.
    Args:
        question (object): An object containing the question text and its
            associated metadata. Expected to have attributes
            `question` (str) and `question_id` (str).
        k (int): The number of top results to retrieve.
        bm25 (BM25Okapi): An instance of the BM25Okapi class used for
            ranking text chunks.
        chunks (list): A list of text chunks to search through. Each
            chunk is expected to have attributes `file_path` (str),
            `start` (int), and `end` (int).
    Returns:
        MinimalSearchResults: An object containing the question ID,
        the question text, and a list of retrieved sources. Each source
        includes the file path and the start and end character
        indices of the relevant chunk.
    """

    token_query = tokenize(question.question)
    scores = bm25.get_scores(token_query)
    for i, chunk in enumerate(chunks):
        if chunk.file_path.suffix in DOC_EXTENSIONS:
            scores[i] *= DOC_BOOST
    top_chunks = [chunks[i] for i in np.argsort(scores)[::-1][:k]]
    src = []

    for chunk in top_chunks:
        src.append(MinimalSource(
            file_path=str(chunk.file_path),
            first_character_index=chunk.start,
            last_character_index=chunk.end,
        ))

    return MinimalSearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=src,
    )


def search(query: str, k: int) -> None:
    """
    Perform a search operation using the given query and return the
    top-k results. This function loads a pre-built BM25 index and document
    chunks from the specified directory, processes the query as an
    `UnansweredQuestion`, and retrieves the top-k search results.
    The results are then wrapped in a `StudentSearchResults` object
    and printed in JSON format.
    Args:
        query (str): The search query string to be processed.
        k (int): The number of top results to retrieve.
    Returns:
        None: This function does not return a value; it prints the
        search results.
    """

    if not valid_query(query) or not valid_positive_int(k, "k"):
        return None

    try:
        (bm25, chunks) = load_index(Path("data/processed"))
    except FileNotFoundError:
        print("Error: index not found. Run 'index' first.")
        return None

    question = UnansweredQuestion(question=query)
    result = search_one(question, k, bm25, chunks)

    student_result = StudentSearchResults(search_results=[result], k=k)
    print(student_result.model_dump_json(indent=2))


def run_search_on_dataset(dataset_path: Path,
                          k: int) -> StudentSearchResults | None:
    """Run the search on every question of a dataset file.

    Args:
        dataset_path: Path of the JSON dataset (``RagDataset`` format).
        k: Number of top sources to retrieve per question.

    Returns:
        The search results of all the questions, in dataset order.
    """

    try:
        dataset = RagDataset.model_validate_json(dataset_path.read_text())
    except FileNotFoundError:
        print(f"Error: dataset file not found: {dataset_path}")
        return None
    except pydantic.ValidationError as e:
        print(f"Error: invalid dataset JSON: {e}")
        return None

    try:
        (bm25, chunks) = load_index(Path("data/processed"))
    except FileNotFoundError:
        print("Error: index not found. Run 'index' first.")
        return None

    all_results = []
    for question in tqdm(dataset.rag_questions, desc="Searching"):
        all_results.append(search_one(question, k, bm25, chunks))

    return StudentSearchResults(search_results=all_results, k=k)


def save_search_results(results: StudentSearchResults, dataset_path: Path,
                        save_directory: Path) -> None:
    """Write the search results as JSON, named after the dataset file.

    Args:
        results: The search results to save.
        dataset_path: Path of the input dataset; its file name is reused.
        save_directory: Directory to write into; created if missing.
    """

    save_directory.mkdir(parents=True, exist_ok=True)
    output_path = save_directory / dataset_path.name
    output_path.write_text(results.model_dump_json(indent=2))

    print(f"Saved student_search_results to {output_path}")


def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
    """Search every question of a dataset and save the results as JSON.

    Args:
        dataset_path: Path of the JSON dataset to read.
        k: Number of top sources to retrieve per question.
        save_directory: Directory where the results file is written.
    """

    if not valid_positive_int(k, "k"):
        return None

    results = run_search_on_dataset(Path(dataset_path), k)
    if results is None:
        return None

    save_search_results(results, Path(dataset_path), Path(save_directory))
