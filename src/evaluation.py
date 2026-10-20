"""Recall@k of search results against a ground-truth dataset."""

from typing import Dict, List

from src.models import AnsweredQuestion, MinimalSource, StudentSearchResults

MIN_IOU = 0.05


def overlaps(a: MinimalSource, b: MinimalSource) -> bool:
    """Tell whether two sources are in the same file with IoU > MIN_IOU.

    Args:
        a: First source.
        b: Second source.

    Returns:
        True when they match.
    """
    if a.file_path != b.file_path:
        return False
    inter = (min(a.last_character_index, b.last_character_index)
             - max(a.first_character_index, b.first_character_index))
    union = (max(a.last_character_index, b.last_character_index)
             - min(a.first_character_index, b.first_character_index))
    return inter > 0 and inter / union > MIN_IOU


def recall_at_k(results: StudentSearchResults,
                truth: List[AnsweredQuestion], k: int) -> float:
    """Average share of correct sources found in the top-k results.

    Args:
        results: Output of search_dataset.
        truth: Questions with their reference sources.
        k: Number of results considered per question.

    Returns:
        Recall between 0 and 1, over the questions present in truth.
    """
    found: Dict[str, List[MinimalSource]] = {
        r.question_id: r.retrieved_sources[:k]
        for r in results.search_results}
    scores = [
        sum(any(overlaps(g, s) for g in found.get(q.question_id, []))
            for s in q.sources) / len(q.sources)
        for q in truth if q.sources]
    return sum(scores) / len(scores) if scores else 0.0
