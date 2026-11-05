import uuid
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """
    MinimalSource is a data model that represents a source of information with
    its file path and the range of character indices that are relevant
    within the file.
    Attributes:
        file_path (str): The path to the file containing the source
            information.
        first_character_index (int): The index of the first relevant character
            in the file.
        last_character_index (int): The index of the last relevant character
            in the file.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    Represents an unanswered question with a unique identifier.
    Attributes:
        question_id (str): A unique identifier for the question,
            automatically generated as a UUID string by default.
        question (str): The text of the question.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    AnsweredQuestion represents a question that has been answered.
    This class extends the UnansweredQuestion class by adding the
        following attributes:
    - sources (list[MinimalSource]): A list of sources that were used
        to answer the question.
    - answer (str): The answer to the question.
    Attributes:
        sources (list[MinimalSource]): A list of sources providing context
                or evidence for the answer.
        answer (str): The text of the answer to the question.
    """

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    RagDataset is a data model representing a collection of questions,
    which can either be answered or unanswered.
    Attributes:
        rag_questions (list[AnsweredQuestion | UnansweredQuestion]):
            A list containing instances of either AnsweredQuestion or
            UnansweredQuestion, representing the questions in the dataset.
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    MinimalSearchResults is a data model representing the results of a minimal
        search operation.
    Attributes:
        question_id (str): A unique identifier for the question.
        question (str): The text of the question being searched.
        retrieved_sources (list[MinimalSource]): A list of sources retrieved
                as a result of the search.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    MinimalAnswer class extends the MinimalSearchResults class and represent
        a minimal structure for an answer in the context of search results.
    Attributes:
        answer (str): The answer text associated with the search result.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """
    StudentSearchResults is a model representing the results of a
        student search.
    Attributes:
        search_results (list[MinimalSearchResults]): A list of minimal search
            result objects.
        k (int): An integer representing the number of search results or a
            specific parameter related to the search.
    """

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    StudentSearchResultsAndAnswer is a data model that represents the results
            of a student search query
    along with the number of top results (k) to be considered.
    Attributes:
        search_results (list[MinimalAnswer]): A list of MinimalAnswer
                objects representing the search results.
        k (int): The number of top results to be considered.
    """

    search_results: list[MinimalAnswer]
    k: int
