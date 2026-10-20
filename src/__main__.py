"""Command-line interface for the RAG project."""

import os
from typing import Type, TypeVar

import fire
from pydantic import BaseModel
from tqdm import tqdm

from src.evaluation import recall_at_k
from src.generator import Generator
from src.indexer import build_index
from src.models import (AnsweredQuestion, MinimalAnswer, MinimalSearchResults,
                        RagDataset, StudentSearchResults,
                        StudentSearchResultsAndAnswer)
from src.retriever import Retriever

Model = TypeVar("Model", bound=BaseModel)


def load(path: str, model: Type[Model]) -> Model:
    """Read and validate a JSON file.

    Args:
        path: JSON file path.
        model: Pydantic model the file must follow.

    Returns:
        The validated data.
    """
    with open(path, encoding="utf-8") as f:
        return model.model_validate_json(f.read())


def save(data: BaseModel, save_directory: str, input_path: str) -> str:
    """Write data as JSON, named after the input file.

    Args:
        data: Model to write.
        save_directory: Output directory, created if missing.
        input_path: File whose name the output reuses.

    Returns:
        The written path.
    """
    os.makedirs(save_directory, exist_ok=True)
    path = os.path.join(save_directory, os.path.basename(input_path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(data.model_dump_json(indent=2))
    return path


def valid_k(k: object) -> bool:
    """Check k and print an error when it is not a positive integer.

    Args:
        k: Value given on the command line.

    Returns:
        True when k is usable.
    """
    if isinstance(k, int) and not isinstance(k, bool) and k > 0:
        return True
    print("Error: --k must be a positive integer")
    return False


class Cli:
    """Commands exposed through Python Fire."""

    def index(self, max_chunk_size: int = 2000, raw_dir: str = "data/raw",
              processed_dir: str = "data/processed") -> None:
        """Build the index from data/raw/ into data/processed/.

        Args:
            max_chunk_size: Maximum size of a chunk in characters.
            raw_dir: Directory holding the corpus.
            processed_dir: Directory where the index is saved.
        """
        if (not isinstance(max_chunk_size, int)
                or not 0 < max_chunk_size <= 2000):
            print("Error: --max_chunk_size must be an integer in [1, 2000]")
            return
        try:
            count = build_index(raw_dir, processed_dir, max_chunk_size)
        except (OSError, ValueError) as e:
            print(f"Error: {e}")
            return
        print(f"Ingestion complete! {count} chunks saved under "
              f"{processed_dir}/")

    def search(self, query: str = "", k: int = 10,
               processed_dir: str = "data/processed") -> None:
        """Print the top-k sources for a single query.

        Args:
            query: The question to search for.
            k: Number of sources to return.
            processed_dir: Directory holding the index.
        """
        if not valid_k(k):
            return
        try:
            sources = Retriever(processed_dir).search([str(query)], k)[0]
        except (OSError, ValueError) as e:
            print(f"Error: {e}")
            return
        result = MinimalSearchResults(question_id="query",
                                      question=str(query),
                                      retrieved_sources=sources)
        print(result.model_dump_json(indent=2))

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results",
                       processed_dir: str = "data/processed") -> None:
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: RagDataset JSON file.
            k: Number of sources per question.
            save_directory: Output directory.
            processed_dir: Directory holding the index.
        """
        if not valid_k(k):
            return
        try:
            questions = load(str(dataset_path), RagDataset).rag_questions
            retriever = Retriever(processed_dir)
            batch = 50
            found = []
            for start in tqdm(range(0, len(questions), batch),
                              desc="Searching"):
                found += retriever.search(
                    [q.question for q in questions[start:start + batch]], k)
            results = StudentSearchResults(k=k, search_results=[
                MinimalSearchResults(question_id=q.question_id,
                                     question=q.question,
                                     retrieved_sources=sources)
                for q, sources in zip(questions, found)])
            path = save(results, str(save_directory), str(dataset_path))
        except (OSError, ValueError) as e:
            print(f"Error: {e}")
            return
        print(f"Saved student_search_results to {path}")

    def answer(self, query: str = "", k: int = 10,
               processed_dir: str = "data/processed",
               model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Answer a single query from the retrieved sources.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve.
            processed_dir: Directory holding the index.
            model_name: Hugging Face model id.
        """
        if not valid_k(k):
            return
        if not str(query).strip():
            print("Error: the query is empty")
            return
        try:
            sources = Retriever(processed_dir).search([str(query)], k)[0]
            text = Generator(model_name).answer(str(query), sources)
        except (OSError, ValueError) as e:
            print(f"Error: {e}")
            return
        result = MinimalAnswer(question_id="query", question=str(query),
                               retrieved_sources=sources, answer=text)
        print(result.model_dump_json(indent=2))

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str = (
                           "data/output/search_results_and_answer"),
                       model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Generate an answer for every question of a search results file.

        Args:
            student_search_results_path: Output of search_dataset.
            save_directory: Output directory.
            model_name: Hugging Face model id.
        """
        path = str(student_search_results_path)
        try:
            results = load(path, StudentSearchResults)
            print(f"Loaded {len(results.search_results)} questions")
            generator = Generator(model_name)
            answers = [
                MinimalAnswer(**r.model_dump(), answer=generator.answer(
                    r.question, r.retrieved_sources))
                for r in tqdm(results.search_results, desc="Answering")]
            out = save(StudentSearchResultsAndAnswer(
                k=results.k, search_results=answers), str(save_directory),
                path)
        except (OSError, ValueError) as e:
            print(f"Error: {e}")
            return
        print(f"Saved student_search_results_and_answer to {out}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print recall@k of search results against answered questions.

        Args:
            student_search_results_path: Output of search_dataset.
            dataset_path: AnsweredQuestions dataset.
        """
        try:
            results = load(str(student_search_results_path),
                           StudentSearchResults)
            dataset = load(str(dataset_path), RagDataset)
        except (OSError, ValueError) as e:
            print(f"Error: {e}")
            return
        truth = [q for q in dataset.rag_questions
                 if isinstance(q, AnsweredQuestion)]
        if not truth:
            print("Error: the dataset has no answered question")
            return
        ks = [k for k in (1, 3, 5, 10) if k <= results.k] or [results.k]
        print("  ".join(f"Recall@{k}: {recall_at_k(results, truth, k):.3f}"
                        for k in ks))


if __name__ == "__main__":
    fire.Fire(Cli)
