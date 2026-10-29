SET ?= docs
K ?= 10
CHUNK ?= 2000
Q ?= How to load a LoRA adapter?
DATASET = data/datasets/UnansweredQuestions/dataset_$(SET)_public.json
TRUTH = data/datasets/AnsweredQuestions/dataset_$(SET)_public.json
RESULTS = data/output/search_results/UnansweredQuestions
ANSWERS = data/output/search_results_and_answer/UnansweredQuestions
VLLM_ZIP ?= vllm-0.10.1.zip
DATASETS_ZIP ?= datasets_public.zip

install:
	uv sync

# Extract the subject attachments into data/raw/ and data/datasets/.
data:
	mkdir -p data/raw data/datasets
	unzip -qo "$(VLLM_ZIP)" -d data/raw
	rm -rf data/.tmp && unzip -qo "$(DATASETS_ZIP)" -d data/.tmp
	cp -r data/.tmp/datasets_public/public/. data/datasets/
	rm -rf data/.tmp

run:
	uv run python -m src index

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

index:
	uv run python -m src index --max_chunk_size $(CHUNK)

search:
	uv run python -m src search "$(Q)" --k $(K)

search_dataset:
	uv run python -m src search_dataset --dataset_path $(DATASET) --k $(K) --save_directory $(RESULTS)

evaluate:
	uv run python -m src evaluate --student_search_results_path $(RESULTS)/dataset_$(SET)_public.json --dataset_path $(TRUTH)

answer:
	uv run python -m src answer "$(Q)" --k $(K)

answer_dataset:
	uv run python -m src answer_dataset --student_search_results_path $(RESULTS)/dataset_$(SET)_public.json --save_directory $(ANSWERS)

.PHONY: install data run debug clean lint lint-strict index search search_dataset evaluate answer answer_dataset