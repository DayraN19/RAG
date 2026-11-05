from .models import MinimalSource, RagDataset, AnsweredQuestion
from .models import StudentSearchResults
from pathlib import Path
import pydantic


def overlaps_enough(ref: MinimalSource, retri: MinimalSource,
                    threshold: float = 0.05) -> bool:
    """
    Determines whether two source segments overlap sufficiently based on a
        given threshold.
    Args:
        ref (MinimalSource): The reference source segment, containing file
            path and character indices.
        retrieved (MinimalSource): The retrieved source segment, containing
            file path and character indices.
        threshold (float, optional): The minimum overlap ratio required for
        the segment to be considered overlapping enough. Defaults to 0.05.
    Returns:
        bool: True if the overlap ratio between the two segments is greater
        than or equal to the threshold, and they belong to the same file.
        False otherwise.
    Notes:
        - The overlap ratio is calculated as the size of the intersection
        divided by the size of the union of the two segments.
        - If the segments are from different files or have no intersection,
                the function returns False.
    """

    if ref.file_path != retri.file_path:
        return False

    inter_start = max(ref.first_character_index,
                      retri.first_character_index)
    inter_end = min(ref.last_character_index, retri.last_character_index)
    intersection = max(0, inter_end - inter_start)

    if intersection == 0:
        return False

    ref_size = ref.last_character_index - ref.first_character_index
    retrieved_size = retri.last_character_index - retri.first_character_index
    union = ref_size + retrieved_size - intersection

    return (intersection / union) >= threshold


def recall_at_k(ref_sources: list[MinimalSource],
                retrieved_sources: list[MinimalSource],
                k: int) -> float | None:
    """
    Calculate the recall at rank k for a set of reference sources and
        retrieved sources.
    Recall at k measures the proportion of relevant items (reference sources)
        that are found
    within the top k retrieved items. It is a metric commonly used in
        information retrieval and recommendation systems.
    Args:
        ref_sources (list[MinimalSource]): A list of reference sources
            representing the ground truth.
        retrieved_sources (list[MinimalSource]): A list of retrieved sources
            to evaluate against the reference sources.
        k (int): The rank cutoff for evaluation. Only the top k
            retrieved sources are considered.
    Returns:
        float | None: The recall at k as a float value between 0 and 1,
            or None if the reference sources list is empty.
    Notes:
        - The function assumes the existence of an `overlaps_enough` function
            that determines if two sources are sufficiently similar
            or overlapping.
        - If `ref_sources` is empty, the function returns None as
        recall cannot be calculated.
    """

    if not ref_sources:
        return None

    top_k = retrieved_sources[:k]
    found = 0
    for ref in ref_sources:
        for r in top_k:
            if overlaps_enough(ref, r):
                found += 1
                break

    return found / len(ref_sources)


def load_reference(dataset_path: Path) -> dict[str, list[MinimalSource]]:
    """
    Load the reference data from a dataset file.
    This function reads a JSON file containing a dataset, validates its
    structure, and extracts the reference sources for each answered question.
    The references are returned as a dictionary where the keys are question
    IDs and the values are lists of `MinimalSource` objects.
    Args:
        dataset_path (Path): The file path to the dataset JSON file.
    Returns:
        dict[str, list[MinimalSource]]: A dictionary mapping question IDs to
        their corresponding reference sources.
    Raises:
        ValidationError: If the dataset JSON file does not conform to
        the expected schema for `RagDataset`.
    """

    dataset = RagDataset.model_validate_json(dataset_path.read_text())
    truth = {}

    for question in dataset.rag_questions:
        if isinstance(question, AnsweredQuestion):
            truth[question.question_id] = question.sources

    return truth


def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """
    Evaluate the recall performance of a student's search results against a
        reference dataset.
    Args:
        student_search_results_path (str): Path to the JSON file containing the
                student's search results.
        dataset_path (str): Path to the reference dataset file.
    Returns:
        None
    This function validates the student's search results JSON file and the
    reference dataset. It calculates recall scores at various k-values
    (1, 3, 5, 10) for each question in the dataset where there is a match
    between the student's results and the reference dataset. The function
    prints the evaluation results, including the average recall scores
    for each k-value.
    Exceptions:
        - Prints an error message and exits if the student's search results
            file or the reference dataset file is not found.
        - Prints an error message and exits if the student's search results
            JSON file is invalid.
        - Prints an error message and exits if there are no matching
            questions between the student's search results and the
            reference dataset.
    """

    try:
        student = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text()
        )
        truth = load_reference(Path(dataset_path))
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return
    except pydantic.ValidationError as e:
        print(f"Error: invalid JSON: {e}")
        return

    k_values = [k for k in [1, 3, 5, 10] if k <= student.k]

    scores: dict[int, list[float]] = {k: [] for k in k_values}
    valued_question = 0

    for result in student.search_results:
        ref_sources = truth.get(result.question_id)
        if not ref_sources:
            continue

        valued_question += 1
        for k in k_values:
            score = recall_at_k(ref_sources, result.retrieved_sources, k)
            assert score is not None  # ref_sources non-empty here
            scores[k].append(score)

    if valued_question == 0:
        print("Error: no matching questions between the search"
              "results and the reference dataset")
        return

    print("Student data is valid: True\n")
    print(f"Questions evaluated: {valued_question}\n")
    print("Evaluation Results\n")
    print("=" * 40)

    for k in k_values:
        average = sum(scores[k]) / len(scores[k])
        print(f"Recall@{k}: {average:.3f} ({average*100:.1f}%)")
