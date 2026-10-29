*This project has been created as part of the 42 curriculum by bgranier.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about the
[vLLM](https://github.com/vllm-project/vllm) 0.10.1 codebase.

It goes through the four classic RAG stages:

1. **Indexing**: every `.py`, `.md` and `.txt` file of `data/raw/` is split into
   chunks of at most 2000 characters and stored in a BM25 index under
   `data/processed/`.
2. **Retrieving**: a question is matched against the index and the top-k chunks
   are returned as source locations (`file_path` + character range).
3. **Augmenting**: the retrieved snippets are numbered and placed in the
   prompt, within a fixed context budget.
4. **Generating**: `Qwen/Qwen3-0.6B` reads that context and writes a short,
   grounded answer.

Retrieval quality is measured with recall@k against the provided ground-truth
datasets.

## Instructions

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and a few GB of free
disk space for PyTorch (CPU build) and the model weights.

```bash
make install          # uv sync: installs the dependencies in .venv/
```

Then extract the subject attachments in place: the corpus goes to
`data/raw/vllm-0.10.1/` and the datasets to `data/datasets/`.

```bash
make data VLLM_ZIP=/path/to/vllm-0.10.1.zip DATASETS_ZIP=/path/to/datasets_public.zip
# or copy both zips to the repository root and run: make data
```

`data/processed/` and `data/output/` are created by the commands themselves.

Every command runs as `uv run python -m src <command> [options]`:

| Command | What it does |
|---|---|
| `index --max_chunk_size 2000` | Chunks `data/raw/` and saves the index in `data/processed/` |
| `search "<query>" --k 10` | Prints the top-k sources for one question |
| `search_dataset --dataset_path <file> --k 10 --save_directory <dir>` | Searches a whole dataset, writes a `StudentSearchResults` JSON |
| `answer "<query>" --k 10` | Retrieves and answers one question |
| `answer_dataset --student_search_results_path <file> --save_directory <dir>` | Answers every question of a search results file |
| `evaluate --student_search_results_path <file> --dataset_path <file>` | Prints recall@1/3/5/10 against an AnsweredQuestions dataset |

Optional flags: `--raw_dir`, `--processed_dir` (index location) and `--model_name`
(any Hugging Face causal LM, `Qwen/Qwen3-0.6B` by default).

Makefile rules: `install`, `run` (builds the index), `debug` (runs under `pdb`),
`clean`, `lint`, `lint-strict`, plus shortcuts such as
`make search_dataset SET=code K=10`.

## Example usage

```bash
# 1. Build the index (about 15 s)
uv run python -m src index --max_chunk_size 2000

# 2. Search a dataset; scope the output directory by dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 3. Check recall with the provided moulinette
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000

# 4. Generate answers from the search results
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions

# Single questions
uv run python -m src search "How do I enable LoRA adapters?" --k 5
uv run python -m src answer "How do I enable LoRA adapters?" --k 5
```

## System architecture

```
data/raw/ ──index──► data/processed/ (bm25/ + chunks.json)
                         │
question ──search──► Retriever ──► top-k MinimalSource ──► search results JSON
                                         │
                                         └──answer──► Generator (Qwen3-0.6B) ──► answer JSON
```

| Module | Role |
|---|---|
| `src/models.py` | Pydantic models exchanged between stages (`MinimalSource`, `RagDataset`, `StudentSearchResults`, ...) |
| `src/indexer.py` | Chunking, tokenization and BM25 index construction |
| `src/retriever.py` | Loads the index once and runs batched top-k searches |
| `src/generator.py` | Builds the prompt from the retrieved snippets and runs the model |
| `src/evaluation.py` | Recall@k with the same IoU rule as the moulinette |
| `src/__main__.py` | Python Fire CLI: input validation, error handling, JSON I/O, progress bars |

`data/processed/bm25/` holds the BM25 matrix and vocabulary. `chunks.json` holds
the chunk spans in the same order, so document `i` of the BM25 index is
`chunks.json[i]`.

## Chunking strategy

There are two strategies, both in `src/indexer.py`:

- **Python code**: cut before every top-level `def`, `async def`, `class` or
  decorator (regex anchored at column 0). Methods stay with their class.
- **Markdown / text**: cut before every heading (`#` to `######`).

Then:

1. Any section longer than `max_chunk_size` is split again, on the last newline
   that fits (or at exactly `max_chunk_size` for a line with no newline).
2. Small neighbouring sections are merged while the chunk stays within
   `max_chunk_size`. A lone import or heading is a weak chunk on its own.

The chunks are contiguous, cover the whole file and are never wider than
`max_chunk_size` (2000 by default, the moulinette limit). Whitespace-only chunks
are dropped. Indices are Python character offsets in the file.

## Retrieval method

The ranking uses **BM25** through the [`bm25s`](https://github.com/xhluca/bm25s)
library (Lucene variant, `k1 = 1.5`, `b = 0.75`):

```
score(q, d) = Σ IDF(t) · tf(t, d) · (k1 + 1) / (tf(t, d) + k1 · (1 − b + b · |d| / avgdl))
             t∈q
```

Rare terms weigh more (IDF), repeated terms saturate (`k1`), and long chunks are
normalized (`b`).

Tokenization is the same for chunks and queries:

- lowercase, English stopwords removed, Snowball stemming (`loading` → `load`);
- identifiers kept **and** split: the indexed text is the original text followed
  by a copy where `snake_case` and `camelCase` are broken into words.
  `max_model_len` matches both a verbatim identifier and "max model length";
- the file path is prepended to each chunk's indexed text, because path words
  (`docs/features/lora.md`) often name the topic.

At query time, unknown tokens are dropped. An empty or nonsensical query
therefore returns no result instead of failing. Chunks with a score of 0 are not
returned. All queries of a dataset are scored in batches.

## Performance analysis

Measured with the provided moulinette (`--k 10 --max_context_length 2000`) on the
public datasets (100 docs questions, 99 code questions):

| `max_chunk_size` | Chunks | Docs R@1 | Docs R@3 | Docs R@5 | Docs R@10 | Code R@1 | Code R@3 | Code R@5 | Code R@10 |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 46 862 | 0.64 | 0.77 | 0.81 | 0.84 | 0.45 | 0.64 | 0.76 | 0.83 |
| 1000 | 24 272 | 0.66 | 0.80 | 0.85 | 0.87 | 0.52 | 0.65 | 0.69 | 0.80 |
| **2000 (default)** | **12 865** | 0.64 | 0.84 | **0.84** | 0.87 | 0.45 | 0.72 | **0.77** | 0.81 |

Required: 80 % recall@5 on docs and 50 % on code. The default size passes both.

Smaller chunks are more precise, but each one is less likely to overlap the
reference span, so recall@3 and recall@5 drop. 2000 gives the best recall@5
overall and also the fewest chunks, which makes the index faster.

| Operation | Time (CPU) | Limit |
|---|---|---|
| Indexing the whole corpus | ~15 s | 5 min |
| `search_dataset` (100 questions, including index loading) | ~4 s | 90 s for 200 questions |
| `answer` on a single question (including model loading) | ~40 s | none |

## Design decisions

- **BM25 over TF-IDF**: same idea, but with term-frequency saturation and length
  normalization, which matter with chunks of very different sizes.
- **`bm25s`**: a sparse-matrix BM25 in numpy. It indexes the corpus in seconds,
  answers batched queries in milliseconds and saves to plain files.
- **Same tokenizer on both sides**: one `tokenize()` function is shared by the
  indexer and the retriever, so tokens always match.
- **Retriever as a class**: the index is loaded once and reused for every
  question of a dataset.
- **Prompt strategy**: the system prompt asks the model to use only the numbered
  snippets, answer in 1 to 4 sentences, name exact functions or options, and say
  it does not know otherwise. The context is capped at 8000 characters, keeping
  the best-ranked snippets first. Qwen3's thinking mode is disabled
  (`enable_thinking=False`) so all 200 new tokens go to the answer. Greedy
  decoding makes answers reproducible. With no retrieved snippet, the model is
  not called at all.
- **Robust CLI**: inputs are checked before use (`k` must be a positive integer,
  `max_chunk_size` must be in [1, 2000], the query must not be empty). Missing
  files and malformed JSON are reported as one-line errors: pydantic's
  `ValidationError` is a `ValueError`, so `except (OSError, ValueError)` covers
  every I/O and parsing failure.
- **CPU-only PyTorch**: `pyproject.toml` pulls torch from the CPU wheel index,
  because campus machines have no GPU and CUDA wheels weigh gigabytes.

## Challenges faced

- **Questions and code use different words.** Splitting identifiers, stemming and
  indexing the file path closed most of the gap without semantic embeddings.
- **The 2000-character limit.** A single over-long source invalidates the whole
  output. The splitter guarantees the bound, and `index` refuses
  `max_chunk_size > 2000`.
- **Exact paths.** The grader compares `file_path` verbatim. Paths are kept
  exactly as walked from `data/raw/` (`data/raw/vllm-0.10.1/...`).
- **A small model on CPU.** Qwen3-0.6B is slow and drifts with long contexts. A
  bounded context, short answers and disabled thinking keep generation usable.
- **Degenerate inputs.** Empty or unknown-word queries, `k=0`, a missing index
  and broken JSON all had to end with a clear message, never a traceback.

## Resources

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020), https://arxiv.org/abs/2005.11401
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, chapters 6 and 11, https://nlp.stanford.edu/IR-book/
- `bm25s` documentation: https://github.com/xhluca/bm25s
- Hugging Face Transformers, chat templates and generation: https://huggingface.co/docs/transformers
- Qwen3-0.6B model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Python Fire: https://github.com/google/python-fire
- Pydantic: https://docs.pydantic.dev

### AI usage

Claude (Anthropic) was used to:

- review the project against the subject and check the requirements;
- test edge cases of the CLI;
- explain the code and the BM25 / recall@k concepts;
- draft this README from the code and the measured results.

All code was reviewed and understood before submission.
