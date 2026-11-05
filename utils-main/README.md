*This project has been created as part of the 42 curriculum by thmaille.*

# RAG against the machine

## Description

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that answers
natural-language questions about a codebase — the [vLLM](https://github.com/vllm-project/vllm)
repository — without retraining or fine-tuning any model.

Instead of relying on a language model's frozen training data, the system:
1. **Indexes** the codebase into searchable chunks (`index`),
2. **Retrieves** the most relevant chunks for a question using a lexical search algorithm
   (`search` / `search_dataset`),
3. **Generates** a grounded, natural-language answer from those chunks with a small local
   LLM (`answer` / `answer_dataset`),
4. **Evaluates** retrieval quality against a ground-truth dataset with a recall@k metric
   (`evaluate`).

The goal is to retrieve only the two or three snippets that actually answer a question out
of thousands of files, then force the model to answer strictly from that evidence rather
than from its own (possibly outdated or hallucinated) knowledge.

## Instructions

### Requirements
- Python 3.10+ (developed and tested on 3.14)
- [`uv`](https://docs.astral.sh/uv/) as the project/package manager
- The vLLM corpus available under `data/raw/`

### Installation
```bash
make install
```
This runs `uv sync` and installs the deep-learning stack (`torch`, `transformers`) needed
to run `Qwen/Qwen3-0.6B` locally.

### Running
```bash
make run          # equivalent to: uv run python -m src
make debug        # same, but through Python's pdb debugger
```

### Code quality
```bash
make lint         # flake8 + mypy (mandatory flags)
```

### Cleaning
```bash
make clean         # removes __pycache__, .mypy_cache, and the generated
                    # data/processed/ (index) and data/output/ (results) directories
```

## System architecture

The pipeline is split into one module per stage, under `src/`:

| File | Role |
|---|---|
| `__main__.py` | CLI entry point, exposes every command through **Python Fire** |
| `models.py` | Pydantic models shared between all stages (validation + JSON I/O) |
| `file_chunker.py` | Pure chunking logic: splits raw text into `RawChunk`s (Python / Markdown strategies) |
| `file_indexer.py` | Discovers files under `data/raw/`, chunks them, builds and persists the BM25 index |
| `file_searcher.py` | Retrieval: BM25 lookup for a single query or a whole dataset |
| `file_generation_batch.py` | Augmentation + generation: builds a token-budgeted context and calls `Qwen/Qwen3-0.6B` |
| `file_evaluate.py` | Self-evaluation: computes recall@k against a ground-truth dataset |

Data flows strictly left to right and never backwards:

```
data/raw/ ──► file_indexer (chunk + BM25) ──► data/processed/bm25_index.pkl
                                                        │
question ──► file_searcher.search_one ──────────────────┘
                   │
                   ▼
         MinimalSource[] (file_path + character range, no raw text stored)
                   │
                   ▼
       file_generation_batch (read text back from data/raw/,
                   build context, prompt Qwen3-0.6B)
                   │
                   ▼
              final answer (JSON)
```

A deliberate design choice: retrieval results only carry a `file_path` and a character
range, never the extracted text itself. The original file under `data/raw/` stays the
single source of truth, which keeps every intermediate JSON file small and avoids any
risk of the stored text drifting from the source.

## Chunking strategy

A Python file and a Markdown page do not break apart the same way, so two distinct
strategies are implemented in `file_chunker.py`:

- **Python files** (`chunk_python_file`): parsed with the `ast` module. Each top-level
  `FunctionDef` / `AsyncFunctionDef` / `ClassDef` becomes one chunk, so in the common case
  a function or a class stays a single, semantically complete unit. Line numbers from the
  AST are converted to absolute character offsets via a cumulative per-line offset table.
- **Markdown / text files** (`chunk_markdown_file`): split on Markdown headers
  (`^#{1,6}\s`), so each chunk follows a documentation section rather than an arbitrary
  cut point.
- **Fallback** (`slice_text`): both strategies fall back to fixed-size, non-overlapping
  slicing when a file has a syntax error, has no top-level definitions / headers, or when
  a resulting block (a function, a class, a section) is still larger than
  `max_chunk_size`. In that case the block itself — not the whole file — is cut into
  consecutive fixed-size slices that, put back together, still cover it exactly: no text
  is lost, only the block's semantic coherence as a single retrievable unit, and only for
  that specific oversized block. This is a deliberate trade-off against a hard constraint:
  the grading moulinette rejects any source wider than `max_context_length`, so a single
  over-long chunk would invalidate the whole output.

`max_chunk_size` is configurable through `--max_chunk_size` (default **2000 characters**)
and is never exceeded — the grading moulinette rejects any source wider than its
`max_context_length` (2000 characters), so a single over-long chunk would invalidate the
whole output.

## Retrieval method

Retrieval is implemented with **BM25** (`rank_bm25.BM25Okapi`), a classic lexical ranking
algorithm (a refinement of TF-IDF):

- At indexing time, every chunk's text is lowercased and tokenized with `re.findall(r"\w+", ...)`
  and fed into `BM25Okapi` to build the index (`file_indexer.build_4_bm25`).
- At query time, the question is tokenized **the exact same way** (lowercase + `\w+`) —
  consistency between indexing and query tokenization is what makes matching possible.
- `bm25.get_top_n(tokens, chunks, n=k)` ranks chunks by BM25 score and returns the top-`k`.

BM25 combines a saturating term-frequency score (repeating a word doesn't linearly
increase relevance forever) with an inverse-document-frequency weight (rare words count
more) and a document-length normalization. It is a **lexical** method: it matches exact
words, not meaning — a deliberate, simple, fast choice that needs no embedding model and
runs well within the project's time constraints (indexing the full corpus in a few
seconds, far under the 5-minute limit).

The built index is persisted with `pickle` under `data/processed/bm25_index.pkl`
(`save_index` / `load_index`), so it is built once and reused across all searches instead
of being rebuilt on every query.

## Answer generation

`file_generation_batch.py` wraps `Qwen/Qwen3-0.6B` (the mandatory model):

- `build_context` concatenates the retrieved sources' text (read back from `data/raw/`
  using the stored character range) up to a fixed token budget (1000 tokens, `MAX_CONTEXT_TOKENS`), so the
  model's context window is never exceeded.
- `build_messages` wraps the question and context in a system prompt that explicitly
  forbids answering outside the provided context ("say you don't know" rather than
  hallucinate).
- `build_prompt` / `generate_batch` apply the tokenizer's chat template (`enable_thinking=False`, to
  avoid Qwen3 emitting a long `<think>...</think>` reasoning block), then generates
  deterministically (`do_sample=False`) with a 100-token budget.
- `load_model` moves the model to **MPS** (Apple Silicon GPU) when available, falling
  back to CPU otherwise — see Performance analysis below for the measured speedup.

## Performance analysis

Measured on the public datasets (k=10) after indexing the full corpus, with the
project's own `evaluate` command:

| Metric | Value | Requirement | Status |
|---|---|---|---|
| Indexing time (20931 chunks) | ~7 s | ≤ 5 min | ✅ |
| Search throughput (100 questions) | ~10–12 s | ≤ 90 s / 200 questions | ✅ |
| Recall@5 — docs questions | **84.0 %** | ≥ 80 % | ✅ |
| Recall@5 — code questions | **80.8 %** | ≥ 50 % | ✅ |

Full recall curve:

| k | Recall@k — docs | Recall@k — code |
|---|---|---|
| 1 | 59.0 % | 53.5 % |
| 3 | 79.0 % | 69.7 % |
| 5 | 84.0 % | 80.8 % |
| 10 | 87.0 % | 87.9 % |

Identifiers are tokenized both whole and split (snake_case / CamelCase), and the file path
is indexed with each chunk, which is what lifted the code recall.

Lexical matching remains the main limitation: BM25 matches words, not meaning, and some
questions paraphrase an idea ("how do I load a model asynchronously")
while the matching answer lives in an identifier ("`AsyncLLMEngine`") — a known limitation
of keyword matching rather than a bug.

Generation throughput: ~2.2 s/question on Apple Silicon (MPS) versus ~25 s/question on
CPU — roughly an **11x speedup** from moving the model to `mps`, with automatic fallback
to CPU when unavailable.

## Design decisions

- **BM25 over embeddings**: no embedding model to download/run, fast indexing, and more
  than sufficient for keyword-heavy technical documentation and source code.
- **Character-offset sources instead of stored text**: `MinimalSource` only carries
  `file_path` + a character range. This keeps every JSON output small, and the original
  file under `data/raw/` remains the single source of truth — there is no copy of the text
  that could drift out of sync.
- **Two chunking strategies**: splitting on AST nodes / Markdown headers keeps each chunk
  a semantically complete unit (a whole function, a whole section) instead of an arbitrary
  fixed-size cut that could sever a definition in the middle.
- **Index built once, loaded once per command**: `load_index` / `load_model` are only
  called once per CLI invocation, outside any per-question loop — required to meet the
  90-second-for-200-questions constraint.
- **Deterministic generation**: `do_sample=False` makes answers reproducible, useful both
  for debugging and for grading consistency.
- **Fail gracefully, not silently**: every CLI command that reads a file wraps it in
  `try/except` for `FileNotFoundError` / `pydantic.ValidationError` and prints a clear
  message instead of letting an unhandled traceback reach the user.

## Challenges faced

- **Absolute character offsets**: converting AST line numbers (and Markdown header line
  indices) into absolute character offsets required building a cumulative per-line length
  table — a line-based position is not directly usable against character-indexed source
  ranges.
- **Tokenization consistency**: BM25 only works if the same tokenization (lowercasing,
  `\w+` word extraction) is applied at indexing time and at query time; any mismatch
  silently degrades retrieval quality without raising an error.
- **`transformers` chat template subtlety**: `apply_chat_template` returns token ids by
  default (`tokenize=True`); forgetting `tokenize=False` silently breaks the downstream
  tokenizer call with a confusing `TypeError`.
- **Incomplete third-party type stubs**: `mypy --strict` flagged `model.to()`,
  `model.generate()` and `model.eval()` as untyped/mismatched calls — verified with
  `reveal_type` to be genuine gaps in `transformers`' own stubs, not real bugs, and
  resolved with narrowly-scoped `# type: ignore` comments.
- **mypy `exclude` is a regex, not a comma list**: the initial `setup.cfg` config
  (`exclude = .venv, .git, __pycache__, data`) silently failed to exclude anything,
  causing `mypy` to scan the entire vendored vLLM corpus and crash on a duplicate module
  name before even reaching the project's own code.
- **Graceful degradation**: every CLI command had to be hardened against missing files,
  malformed JSON, and empty queries / invalid `k` (`k<=0`, non-integer) — rejected with a
  clear message by `valid_query` / `valid_positive_int` — without crashing — verified by deliberately
  reproducing each failure case.
- **Code recall**: plain `\w+` tokenization missed identifiers; splitting snake_case /
  CamelCase and indexing the file path with each chunk raised code recall@5 above the
  threshold with a comfortable margin.

## Example usage

```bash
# 1. Build the index (once)
uv run python -m src index --max_chunk_size 2000

# 2. Single-query search
uv run python -m src search --query="How to configure the OpenAI server?" --k=5

# 3. Search a whole dataset
uv run python -m src search_dataset \
  --dataset_path=data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k=10 \
  --save_directory=data/output/search_results/UnansweredQuestions

# 4. Evaluate retrieval quality against the ground truth
uv run python -m src evaluate \
  --student_search_results_path=data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path=data/datasets/AnsweredQuestions/dataset_docs_public.json

# 5. Single-query answer (retrieval + generation)
uv run python -m src answer --query="What is vLLM?" --k=3

# 6. Answer a whole dataset of already-searched questions
uv run python -m src answer_dataset \
  --student_search_results_path=data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory=data/output/search_results_and_answer/UnansweredQuestions
```

## Resources

### References
- Robertson, S. & Zaragoza, H., *The Probabilistic Relevance Framework: BM25 and Beyond*,
  2009 — the algorithm behind the retrieval stage.
- Lewis, P. et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*,
  2020 — the original RAG paper.
- [`rank_bm25`](https://github.com/dorianbrown/rank_bm25) documentation.
- [Hugging Face `transformers`](https://huggingface.co/docs/transformers) documentation.
- [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) model card.
- [Pydantic](https://docs.pydantic.dev/) documentation.
- [Python Fire](https://github.com/google/python-fire) documentation.

### AI usage
AI (Claude Code) was used throughout this project as a **reviewer and debugging aid**,
not as a code generator: the implementation was written by hand, then reviewed line by
line, with bugs pointed out precisely (file, line, failure scenario) so they could be
understood and fixed personally rather than patched automatically. Concretely, AI was
used to:
- review each module after it was written and flag correctness bugs (e.g. a chained
  assignment silently corrupting data in the recall calculation, a mis-scoped counter
  giving wrong recall values, mismatched `except` clauses hiding the wrong error),
- diagnose `mypy`/`mypy --strict` failures against third-party library stubs
  (`transformers`), verified with `reveal_type` before deciding whether to fix or
  suppress,
- help design and run the end-to-end validation against the real grading moulinette
  (via `moulinette-mac.sh`) and interpret the resulting recall@k figures,
- draft this README from the architecture, decisions and numbers established during the
  project.

Every AI-assisted suggestion was reviewed, tested, and understood before being applied —
ready to be explained and defended during the oral evaluation.
