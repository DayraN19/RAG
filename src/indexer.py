"""Chunk the raw corpus and build a persisted BM25 index."""

import json
import os
import re
from typing import List, Pattern, Tuple

import bm25s
import Stemmer
from tqdm import tqdm

Span = Tuple[int, int]

# A Python file breaks at top-level definitions, a Markdown page at headings.
PY_BOUNDARY: Pattern[str] = re.compile(
    r"^(?:@|def |async def |class )", re.MULTILINE)
MD_BOUNDARY: Pattern[str] = re.compile(r"^#{1,6} ", re.MULTILINE)
EXTENSIONS: Tuple[str, ...] = (".py", ".md", ".txt")
STEMMER = Stemmer.Stemmer("english")
# Tuned on recall@5: narrower chunks rank better, even if 2000 is allowed.
BEST_CHUNK_SIZE = 1500


def split_long(text: str, start: int, end: int, max_size: int) -> List[Span]:
    """Cut text[start:end] into spans of at most max_size characters.

    Args:
        text: Whole file content.
        start: First character of the section.
        end: End (exclusive) of the section.
        max_size: Maximum span width.

    Returns:
        Spans, cut on a newline when possible.
    """
    spans: List[Span] = []
    while end - start > max_size:
        cut = text.rfind("\n", start + 1, start + max_size)
        if cut == -1:
            cut = start + max_size
        spans.append((start, cut))
        start = cut
    spans.append((start, end))
    return spans


def chunk(text: str, boundary: Pattern[str], max_size: int) -> List[Span]:
    """Split text at boundaries, then merge small neighbours up to max_size.

    Args:
        text: Whole file content.
        boundary: Regex whose matches start a new section.
        max_size: Maximum chunk width in characters.

    Returns:
        Contiguous (start, end) spans covering the whole text.
    """
    cuts = (m.start() for m in boundary.finditer(text))
    edges = sorted({0, len(text), *cuts})
    sections: List[Span] = []
    for start, end in zip(edges, edges[1:]):
        sections += split_long(text, start, end, max_size)
    chunks: List[Span] = []
    for start, end in sections:
        if chunks and end - chunks[-1][0] <= max_size:
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))
    return chunks


def to_index_text(text: str) -> str:
    """Add split identifiers so 'max_model_len' also matches 'max model len'.

    Args:
        text: Raw chunk or query text.

    Returns:
        The text followed by its identifier parts.
    """
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).replace("_", " ")
    return f"{text}\n{parts}"


def tokenize(texts: List[str]) -> List[List[str]]:
    """Tokenize chunks or queries the same way for indexing and search.

    Args:
        texts: Texts to tokenize.

    Returns:
        One list of string tokens per text.
    """
    tokens: List[List[str]] = bm25s.tokenize(
        [to_index_text(t) for t in texts], stopwords="en", stemmer=STEMMER,
        return_ids=False, show_progress=False)
    return tokens


def build_index(raw_dir: str, processed_dir: str, max_chunk_size: int) -> int:
    """Chunk every corpus file and save the BM25 index and chunk spans.

    Args:
        raw_dir: Corpus root, e.g. data/raw.
        processed_dir: Output directory, e.g. data/processed.
        max_chunk_size: Maximum chunk width in characters.

    Returns:
        Number of chunks indexed.
    """
    paths = sorted(os.path.join(root, name)
                   for root, _, names in os.walk(raw_dir)
                   for name in names if name.endswith(EXTENSIONS))
    max_chunk_size = min(max_chunk_size, BEST_CHUNK_SIZE)
    spans: List[dict] = []
    texts: List[str] = []
    for path in tqdm(paths, desc="Chunking"):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as e:
            print(f"Skipping {path}: {e}")
            continue
        boundary = PY_BOUNDARY if path.endswith(".py") else MD_BOUNDARY
        for start, end in chunk(text, boundary, max_chunk_size):
            if text[start:end].strip():
                spans.append({"file_path": path,
                              "first_character_index": start,
                              "last_character_index": end})
                # The path is indexed too: its words often name the topic.
                texts.append(f"{path}\n{text[start:end]}")
    if not texts:
        raise ValueError(f"No .py/.md/.txt files found under {raw_dir}")
    retriever = bm25s.BM25()
    retriever.index(tokenize(texts))
    os.makedirs(processed_dir, exist_ok=True)
    retriever.save(os.path.join(processed_dir, "bm25"))
    with open(os.path.join(processed_dir, "chunks.json"), "w") as f:
        json.dump(spans, f)
    return len(spans)
