import ast
import re
from dataclasses import dataclass


@dataclass
class RawChunk:
    """A raw text chunk with its absolute character offsets in the source
        file.
    """

    text: str
    start: int
    end: int


HEADER_PATTERN = re.compile(r"^#{1,6}\s")
SLICE_TARGET_SIZE = 1200


def slice_text(text: str, offset_start: int,
               max_chunk_size: int) -> list[RawChunk]:
    """Split text into non-overlapping chunks, cutting at paragraph or
    line boundaries when possible.

    Each chunk is at most ``min(max_chunk_size, SLICE_TARGET_SIZE)``
    characters; the cut is moved back to the last blank line (or else
    the last newline) found in the second two thirds of the window.

    Args:
        text: The text to split.
        offset_start: Absolute character offset of ``text`` within the
            original file (0 if ``text`` already is the whole file).
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        The list of chunks, skipping any piece that is empty or
        whitespace-only.
    """

    chunk_result: list[RawChunk] = []
    size = min(max_chunk_size, SLICE_TARGET_SIZE)
    pos = 0
    len_txt = len(text)

    while pos < len_txt:
        end = min(pos + size, len_txt)
        if end < len_txt:
            cut = text.rfind("\n\n", pos + size // 3, end)
            if cut == -1:
                cut = text.rfind("\n", pos + size // 3, end)
            if cut != -1:
                end = cut + 1
        part = text[pos:end]

        if part.strip():
            chunk_result.append(RawChunk(text=part, start=offset_start + pos,
                                         end=offset_start + end))

        pos = end

    return chunk_result


def chunk_python_file(data_file: str, max_chunk_size: int) -> list[RawChunk]:
    """Chunk a Python source file along its top-level functions/classes.

    Falls back to fixed-size chunking when the file has a syntax error,
    has no top-level function/class definition, or when a block is
    larger than ``max_chunk_size``.

    Args:
        data_file: The full source code of the Python file.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        The list of chunks covering the file.
    """

    try:
        tree = ast.parse(data_file)
    except SyntaxError:
        return slice_text(data_file, 0, max_chunk_size)

    lines = data_file.splitlines(keepends=True)
    offset_line = [0]
    for line in lines:
        offset_line.append(offset_line[-1] + len(line))

    blocs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            start_line = node.lineno - 1
            end_line = (node.end_lineno if node.end_lineno is not None
                        else start_line + 1)
            start_char = offset_line[start_line]
            end_char = offset_line[end_line]
            blocs.append((start_char, end_char))

    if not blocs:
        return slice_text(data_file, 0, max_chunk_size)

    final_chunks: list[RawChunk] = []
    previous_end = 0
    for start_char, end_char in blocs:
        if start_char > previous_end:
            final_chunks.extend(slice_text(
                data_file[previous_end:start_char], previous_end,
                max_chunk_size))
        bloc_text = data_file[start_char:end_char]
        if len(bloc_text) <= max_chunk_size:
            final_chunks.append(
                RawChunk(text=bloc_text, start=start_char, end=end_char))
        else:
            final_chunks.extend(
                slice_text(bloc_text, start_char, max_chunk_size))
        previous_end = end_char

    if previous_end < len(data_file):
        final_chunks.extend(slice_text(
            data_file[previous_end:], previous_end, max_chunk_size))

    return final_chunks


def chunk_markdown_file(data_file: str, max_chunk_size: int) -> list[RawChunk]:
    """Chunk a Markdown or plain text file along its headers.

    Falls back to fixed-size chunking when the file has no Markdown
    header, or when a section is larger than ``max_chunk_size``.

    Args:
        data_file: The full content of the Markdown/text file.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        The list of chunks covering the file.
    """

    lines = data_file.splitlines(keepends=True)
    offset_line = [0]
    for line in lines:
        offset_line.append(offset_line[-1] + len(line))

    header_indices = [
        i for i, line in enumerate(lines) if HEADER_PATTERN.match(line)
    ]

    if not header_indices:
        return slice_text(data_file, 0, max_chunk_size)

    sections = []
    for pos, start_line in enumerate(header_indices):
        if pos + 1 < len(header_indices):
            end_line = header_indices[pos + 1]
        else:
            end_line = len(lines)
        sections.append((offset_line[start_line], offset_line[end_line]))

    if header_indices[0] > 0:
        sections.insert(0, (0, offset_line[header_indices[0]]))

    final_chunks: list[RawChunk] = []
    for start_char, end_char in sections:
        section_text = data_file[start_char:end_char]
        if len(section_text) <= max_chunk_size:
            final_chunks.append(
                RawChunk(text=section_text, start=start_char, end=end_char))
        else:
            final_chunks.extend(
                slice_text(section_text, start_char, max_chunk_size))

    return final_chunks
