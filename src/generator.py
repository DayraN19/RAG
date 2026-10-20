"""Generate grounded answers with a small local model."""

from typing import Any, Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import MinimalSource

SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase. Use only the numbered "
    "context snippets. Answer in 1 to 4 sentences and name the exact "
    "function, class, option or endpoint when the snippets show it. If the "
    "snippets do not contain the answer, say you do not know.")
# shortcut: fixed character budget (~2-3k tokens), switch to a token count
# if a bigger model or a larger k needs more context.
MAX_CONTEXT_CHARS = 8000


class Generator:
    """Wrap the model and its tokenizer."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load the model on CPU.

        Args:
            model_name: Hugging Face model id.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Any: transformers' type stubs reject .generate() on the auto class.
        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        self.files: Dict[str, str] = {}

    def read(self, source: MinimalSource) -> str:
        """Return the text a source points to.

        Args:
            source: File path and character range.

        Returns:
            The snippet, or an empty string if the file cannot be read.
        """
        if source.file_path not in self.files:
            try:
                with open(source.file_path, encoding="utf-8",
                          errors="replace") as f:
                    self.files[source.file_path] = f.read()
            except OSError:
                self.files[source.file_path] = ""
        text = self.files[source.file_path]
        return text[source.first_character_index:
                    source.last_character_index]

    def answer(self, question: str, sources: List[MinimalSource]) -> str:
        """Answer a question from the given sources.

        Args:
            question: The user question.
            sources: Retrieved sources, best first.

        Returns:
            The generated answer.
        """
        context = ""
        for n, source in enumerate(sources, 1):
            block = f"[{n}] {source.file_path}\n{self.read(source)}\n\n"
            if len(context) + len(block) > MAX_CONTEXT_CHARS:
                break
            context += block
        if not context:
            return "I do not know: no relevant source was found."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context}Question: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=200,
                                     do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return str(text).strip()
