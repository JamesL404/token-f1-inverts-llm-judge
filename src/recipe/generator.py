"""LLM generator interface for C-RAG.

Pluggable backends:
  - DummyLLM: returns the question text (CPU sanity check, no LLM)
  - OpenAILLM: OpenAI / vLLM-OpenAI compatible (lazy import)
  - HFLocalLLM: local HuggingFace causal LM, intended for direct GPU runs

The interface is intentionally minimal so the rest of the C-RAG pipeline
can be tested end-to-end on CPU before any LLM serving is set up.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass
class GenerationCall:
    prompt: str
    completion: str
    prompt_chars: int
    completion_chars: int
    latency_s: float
    model: str
    backend: str


class Generator:
    """Abstract LLM generator."""
    name: str = "abstract"
    model: str = "abstract"

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> GenerationCall:
        raise NotImplementedError


class DummyLLM(Generator):
    """No-op generator that returns a deterministic stub. Useful for CPU smoke tests.

    Two modes:
      - 'echo_question': returns the question text (extracted from the prompt
        with a heuristic regex). Used to verify the recipe scaffolding runs.
      - 'fixed': returns a fixed string.
    """
    name = "dummy"

    def __init__(self, mode: str = "echo_question", fixed_text: str = "no information available"):
        self.model = f"dummy({mode})"
        self.mode = mode
        self.fixed_text = fixed_text

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> GenerationCall:
        t0 = time.perf_counter()
        if self.mode == "echo_question":
            # Pull out the question line from the prompt (heuristic)
            import re
            m = re.search(r"^Question:\s*(.+?)$", prompt, re.MULTILINE)
            completion = m.group(1).strip() if m else "DUMMY"
        else:
            completion = self.fixed_text
        latency = time.perf_counter() - t0
        return GenerationCall(
            prompt=prompt,
            completion=completion,
            prompt_chars=len(prompt),
            completion_chars=len(completion),
            latency_s=latency,
            model=self.model,
            backend="dummy",
        )


class OpenAILLM(Generator):
    """OpenAI / vLLM-OpenAI-compatible backend.

    For local vLLM serving:
      base_url = "http://localhost:8000/v1"
      api_key = "EMPTY"  (vLLM doesn't check)
      model = "Qwen/Qwen2.5-14B-Instruct" (or whatever vllm was launched with)

    For real OpenAI:
      base_url = None (uses default)
      api_key = OPENAI_API_KEY env var
      model = "gpt-4o-mini"
    """
    name = "openai"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-14B-Instruct",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        from openai import OpenAI  # lazy import
        self.model = model
        self._client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
        )

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> GenerationCall:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        completion = (resp.choices[0].message.content or "").strip()
        latency = time.perf_counter() - t0
        return GenerationCall(
            prompt=prompt,
            completion=completion,
            prompt_chars=len(prompt),
            completion_chars=len(completion),
            latency_s=latency,
            model=self.model,
            backend="openai",
        )


class HFLocalLLM(Generator):
    """Local HuggingFace causal LM backend.

    This is the fallback path when vLLM is unavailable but a local model is
    cached and GPUs are available. It is intentionally single-prompt to fit
    the existing CRAG interface.
    """

    name = "hf-local"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype: str = "auto",
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self._torch = torch
        if torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> GenerationCall:
        t0 = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        input_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(input_text, return_tensors="pt")
        device = getattr(self._model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        do_sample = temperature > 0
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        completion = self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        latency = time.perf_counter() - t0
        return GenerationCall(
            prompt=prompt,
            completion=completion,
            prompt_chars=len(prompt),
            completion_chars=len(completion),
            latency_s=latency,
            model=self.model,
            backend="hf-local",
        )


def get_generator(name: str | None = None, **kwargs) -> Generator:
    """Factory by name. Default: DummyLLM (CPU-safe)."""
    name = name or os.getenv("CRAG_GENERATOR", "dummy")
    if name == "dummy":
        return DummyLLM(**kwargs)
    if name == "openai":
        return OpenAILLM(**kwargs)
    if name == "hf-local":
        return HFLocalLLM(**kwargs)
    raise ValueError(f"unknown generator: {name}")
