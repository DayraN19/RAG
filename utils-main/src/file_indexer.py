import re
import pickle
from dataclasses import dataclass
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pathlib import Path
from .file_chunker import RawChunk, chunk_python_file, chunk_markdown_file


@dataclass
class IndexedChunk:
    """A text chunk tied to the file it was extracted from."""

    file_path: Path
    text: str
    start: int
    end: int


WORD_RE = re.compile(r"\w+")
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOP_WORDS = set(
    "a an the of to in on for and or is are was were be been being what "
    "which who whom how why when where does do did can could should would "
    "will shall may might must that this these those it its with by as at "
    "from into about than then there their they them you your we our i me "
    "my he she his her not no yes if else so such any all each some more "
    "most other".split())
SUFFIXES = ("ing", "ed", "es", "s")
BM25_K1 = 0.9
BM25_B = 0.4
PATH_WEIGHT = 2


def stem(token: str) -> str:
    """Strip a common English suffix (light stemming).

    Args:
        token: A lowercase token.

    Returns:
        The token without its suffix, unless it would get too short.
    """

    for suffix in SUFFIXES:
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    """Split text into lowercase stemmed tokens, also splitting snake_case
    and CamelCase identifiers into their parts. Stop words are dropped.

    Args:
        text: The text to tokenize.

    Returns:
        The list of tokens (full words plus their sub-parts).
    """

    tokens: list[str] = []
    for word in WORD_RE.findall(text):
        tokens.append(word.lower())
        parts = [p.lower() for piece in word.split("_")
                 for p in CAMEL_RE.findall(piece)]
        if len(parts) > 1:
            tokens.extend(parts)
    return [stem(t) for t in tokens if t not in STOP_WORDS]


def discover_file(raw_dir: Path) -> list[Path]:
    """Recursively discover the files to index under raw_dir.

    Args:
        raw_dir: Root directory to scan recursively.

    Returns:
        The list of file paths whose extension is one of ``.py``,
        ``.md``, ``.rst`` or ``.txt``.
    """

    valid_extensions = {".py", ".md", ".rst", ".txt"}
    file_to_index: list[Path] = []

    for file_path in raw_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in valid_extensions:
            file_to_index.append(file_path)

    return file_to_index


def chunker_file(file_path: Path, content: str,
                 max_chunk_size: int) -> list[RawChunk]:
    """Chunk a single file's content according to its extension.

    Args:
        file_path: Path of the file, used to pick the chunking
            strategy (Python vs. Markdown/text).
        content: Full text content of the file.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        The list of chunks covering the file.
    """

    if file_path.suffix == ".py":
        return chunk_python_file(content, max_chunk_size)
    else:
        return chunk_markdown_file(content, max_chunk_size)


def build_corpus(raw_dir: Path, max_chunk_size: int) -> list[IndexedChunk]:
    """Discover, read and chunk every file under raw_dir.

    Args:
        raw_dir: Root directory to scan recursively.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        The list of chunks covering the corpus, each tagged with the
        file_path it was extracted from.
    """

    corpus: list[IndexedChunk] = []

    for file_path in tqdm(discover_file(raw_dir), desc="Chunking"):
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            print(f"Skipping {file_path}: {error}")
            continue

        for chunk in chunker_file(file_path, content, max_chunk_size):
            corpus.append(
                IndexedChunk(file_path=file_path, text=chunk.text,
                             start=chunk.start, end=chunk.end)
            )
    return corpus


def build_4_bm25(all_chunk: list[IndexedChunk]) -> BM25Okapi:
    """Build a BM25 index over a list of chunks.

    Args:
        all_chunk: The chunks to index; each chunk's ``text`` is
            lowercased and split into word tokens before indexing.

    Returns:
        The BM25 index built from the tokenized chunks.
    """

    tokenise_all = []

    for chunk in tqdm(all_chunk, desc="Tokenizing"):
        # the path is repeated to weight it more than the chunk body
        paths = " ".join([str(chunk.file_path)] * PATH_WEIGHT)
        tok = tokenize(paths + " " + chunk.text)
        tokenise_all.append(tok)

    bm25 = BM25Okapi(tokenise_all, k1=BM25_K1, b=BM25_B)
    return bm25


def save_index(bm25: BM25Okapi, chunks: list[IndexedChunk],
               processed_dir: Path) -> None:
    """Persist a BM25 index and its chunks to disk.

    Args:
        bm25: The BM25 index built by :func:`build_4_bm25`.
        chunks: The chunks the index was built from, in the same
            order, so a result index can be mapped back to a
            file_path and character range.
        processed_dir: Directory to write the index file into;
            created if it does not exist yet.
    """

    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)


def load_index(processed_dir: Path) -> tuple[BM25Okapi, list[IndexedChunk]]:
    """Load a previously persisted BM25 index and its chunks.

    Args:
        processed_dir: Directory the index was saved into by
            :func:`save_index`.

    Returns:
        The BM25 index and the list of chunks it was built from.
    """

    with open(processed_dir / "bm25_index.pkl", "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]
