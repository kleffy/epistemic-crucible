"""LLM backends + a response cache for the LLM agent.

Backends share one interface: ``generate_batch(system, conversations)`` returns
one completion per conversation. This lets a single agent step and the batched
rollout harness use the same code path. Greedy decoding + an on-disk cache make
runs reproducible and free to repeat.

Backends:
- ``mock``         — deterministic, no network/GPU (tests, CI).
- ``transformers`` — local Hugging Face model on GPU (the primary path).
- ``anthropic``    — Claude API (built but dormant until ANTHROPIC_API_KEY is set).
- ``openai``       — OpenAI / any OpenAI-compatible server (e.g. a local vLLM).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from crucible.utils.logging import get_logger

_log = get_logger(__name__)

Conversation = list[dict]  # list of {"role": "user"|"assistant", "content": str}


@dataclass(frozen=True)
class GenerationRecord:
    """Visible completion plus the effective settings needed to audit it."""

    response: str
    model_id: str
    backend: str
    requested_params: dict
    effective_params: dict
    usage: dict
    finish_reason: str | None
    response_id: str | None
    cache_hit: bool = False

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "model_id": self.model_id,
            "backend": self.backend,
            "requested_params": self.requested_params,
            "effective_params": self.effective_params,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "response_id": self.response_id,
            "cache_hit": self.cache_hit,
        }


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


class ResponseCache:
    """Disk-backed prompt→response cache with explicit acquisition/replay policy."""

    def __init__(
        self,
        path: str | pathlib.Path | None,
        *,
        read_enabled: bool = True,
        write_enabled: bool = True,
        require_hits: bool = False,
    ) -> None:
        if require_hits and not read_enabled:
            raise ValueError("require_hits requires cache reads")
        self.path = pathlib.Path(path) if path else None
        self.read_enabled = read_enabled
        self.write_enabled = write_enabled
        self.require_hits = require_hits
        self._mem: dict[str, str] = {}
        self._records: dict[str, dict] = {}
        if self.read_enabled and self.path and self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec["response"]
                    self._records[rec["key"]] = rec
            _log.info("loaded %d cached responses from %s", len(self._mem), self.path)

    @staticmethod
    def key(model_id: str, system: str, messages: Conversation, gen: dict) -> str:
        blob = json.dumps(
            {"model": model_id, "system": system, "messages": messages, "gen": gen},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        if not self.read_enabled:
            return None
        return self._mem.get(key)

    def get_record(self, key: str) -> dict | None:
        if not self.read_enabled:
            return None
        record = self._records.get(key)
        return dict(record) if record is not None else None

    def put(self, key: str, response: str) -> None:
        self.put_record(key, {"response": response})

    def put_record(self, key: str, record: dict) -> None:
        if not self.write_enabled:
            return
        if key in self._mem:
            return
        response = str(record["response"])
        self._mem[key] = response
        stored = {"key": key, **record, "response": response}
        self._records[key] = stored
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(stored, sort_keys=True, default=str) + "\n")


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class LLMBackend(ABC):
    model_id: str
    gen_params: dict

    def __init__(
        self,
        model_id: str,
        cache: ResponseCache | None = None,
        *,
        cache_key_params: dict[str, Any] | None = None,
        **gen_params,
    ) -> None:
        self.model_id = model_id
        self.cache = cache
        self.gen_params = {"max_new_tokens": 192, "temperature": 0.0, **gen_params}
        self.cache_key_params = cache_key_params

    def generate_batch(
        self,
        system: str,
        conversations: list[Conversation],
        *,
        cache_contexts: list[dict[str, Any] | None] | None = None,
    ) -> list[str]:
        return [
            record.response
            for record in self.generate_batch_records(
                system, conversations, cache_contexts=cache_contexts
            )
        ]

    def generate_batch_records(
        self,
        system: str,
        conversations: list[Conversation],
        *,
        cache_contexts: list[dict[str, Any] | None] | None = None,
    ) -> list[GenerationRecord]:
        """Cache-aware generation with per-completion provenance.

        ``cache_contexts`` deliberately separates experimentally distinct calls
        whose visible prompts are byte-identical. This is required for crossed
        hidden-mechanism cells: sharing one completion there would silently turn
        two measurements into one API call.
        """
        if cache_contexts is None:
            cache_contexts = [None] * len(conversations)
        if len(cache_contexts) != len(conversations):
            raise ValueError("cache_contexts must align one-to-one with conversations")
        results: list[GenerationRecord | None] = [None] * len(conversations)
        misses, miss_idx = [], []
        for i, (conv, context) in enumerate(zip(conversations, cache_contexts, strict=True)):
            key = None
            if self.cache is not None:
                cache_parameters = self.cache_key_params or self.gen_params
                if context is not None:
                    cache_parameters = {
                        "generation": cache_parameters,
                        "experimental_context": context,
                    }
                key = ResponseCache.key(self.model_id, system, conv, cache_parameters)
                hit = self.cache.get_record(key)
                if hit is not None:
                    results[i] = GenerationRecord(
                        response=str(hit["response"]),
                        model_id=str(hit.get("model_id", self.model_id)),
                        backend=str(hit.get("backend", self.backend_name)),
                        requested_params=dict(hit.get("requested_params", self.gen_params)),
                        effective_params=dict(hit.get("effective_params", self.gen_params)),
                        usage=dict(hit.get("usage", {})),
                        finish_reason=hit.get("finish_reason"),
                        response_id=hit.get("response_id"),
                        cache_hit=True,
                    )
                    continue
            misses.append(conv)
            miss_idx.append((i, key))
        if misses and self.cache is not None and self.cache.require_hits:
            raise RuntimeError(
                "artifact replay cache miss: live generation is disabled "
                f"({len(misses)} missing responses)"
            )
        if misses:
            generated = self._generate_batch_records(system, misses)
            for (i, key), record in zip(miss_idx, generated):
                results[i] = record
                if self.cache is not None and key is not None:
                    self.cache.put_record(key, record.to_dict())
        return [
            record
            if record is not None
            else GenerationRecord(
                response="",
                model_id=self.model_id,
                backend=self.backend_name,
                requested_params=dict(self.gen_params),
                effective_params=dict(self.gen_params),
                usage={},
                finish_reason=None,
                response_id=None,
            )
            for record in results
        ]

    @property
    def backend_name(self) -> str:
        return self.__class__.__name__.removesuffix("Backend").lower()

    def _generate_batch_records(
        self,
        system: str,
        conversations: list[Conversation],
    ) -> list[GenerationRecord]:
        return [
            GenerationRecord(
                response=response,
                model_id=self.model_id,
                backend=self.backend_name,
                requested_params=dict(self.gen_params),
                effective_params=dict(self.gen_params),
                usage={},
                finish_reason=None,
                response_id=None,
            )
            for response in self._generate_batch(system, conversations)
        ]

    @abstractmethod
    def _generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]: ...


# ---------------------------------------------------------------------------
# Mock backend (tests / CI)
# ---------------------------------------------------------------------------


class MockBackend(LLMBackend):
    """Deterministic backend. ``policy(system, conversation) -> str`` controls output.

    Default policy returns a WAIT action line.
    """

    def __init__(
        self,
        model_id: str = "mock",
        policy: Callable[[str, Conversation], str] | None = None,
        cache: ResponseCache | None = None,
        **gen_params,
    ) -> None:
        super().__init__(model_id, cache, **gen_params)
        self._policy = policy or (lambda system, conv: 'ACTION: {"kind": "wait", "args": {}}')

    def _generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]:
        return [self._policy(system, conv) for conv in conversations]


class CacheOnlyBackend(LLMBackend):
    """Replay cached generations without constructing or contacting a live model."""

    def _generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]:
        raise RuntimeError(
            "cache-only backend cannot generate; artifact replay requires complete cache hits"
        )


# ---------------------------------------------------------------------------
# Local transformers backend (GPU)
# ---------------------------------------------------------------------------

# Module-level cache so large weights load once and are shared across agents.
_HF_MODELS: dict[tuple, tuple] = {}


class TransformersBackend(LLMBackend):
    def __init__(
        self,
        model_id: str,
        cache: ResponseCache | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 1,
        quantization: str | None = None,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        **gen_params,
    ) -> None:
        super().__init__(
            model_id,
            cache,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            quantization=quantization,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            **gen_params,
        )
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.quantization = quantization
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision or model_revision
        self._tok, self._model = _load_hf_model(
            model_id,
            device,
            dtype,
            quantization,
            model_revision=model_revision,
            tokenizer_revision=self.tokenizer_revision,
        )

    def _generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]:
        return [record.response for record in self._generate_batch_records(system, conversations)]

    def _generate_batch_records(
        self,
        system: str,
        conversations: list[Conversation],
    ) -> list[GenerationRecord]:
        import torch

        records: list[GenerationRecord] = []
        template_kwargs = dict(
            self.gen_params.get("extra_body", {}).get("chat_template_kwargs", {})
        )
        for start in range(0, len(conversations), self.batch_size):
            chunk = conversations[start : start + self.batch_size]
            prompts = [
                self._tok.apply_chat_template(
                    _merge_roles([{"role": "system", "content": system}, *conv]),
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                for conv in chunk
            ]
            enc = self._tok(prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                gen = self._model.generate(
                    **enc,
                    max_new_tokens=self.gen_params["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=self._tok.pad_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1] :]
            responses = self._tok.batch_decode(new_tokens, skip_special_tokens=True)
            prompt_lengths = enc["attention_mask"].sum(dim=1).tolist()
            completion_lengths = [
                int((tokens != self._tok.pad_token_id).sum().item()) for tokens in new_tokens
            ]
            for response, prompt_tokens, completion_tokens in zip(
                responses, prompt_lengths, completion_lengths, strict=True
            ):
                records.append(
                    GenerationRecord(
                        response=response,
                        model_id=self.model_id,
                        backend=self.backend_name,
                        requested_params=dict(self.gen_params),
                        effective_params={
                            "model": self.model_id,
                            "model_revision": self.model_revision,
                            "tokenizer_revision": self.tokenizer_revision,
                            "device": self.device,
                            "dtype": self.dtype,
                            "quantization": self.quantization,
                            "batch_size": self.batch_size,
                            "max_new_tokens": self.gen_params["max_new_tokens"],
                            "temperature": 0.0,
                            "do_sample": False,
                            "chat_template_kwargs": template_kwargs,
                        },
                        usage={
                            "prompt_tokens": int(prompt_tokens),
                            "completion_tokens": completion_tokens,
                            "total_tokens": int(prompt_tokens) + completion_tokens,
                        },
                        finish_reason=(
                            "length"
                            if completion_tokens >= self.gen_params["max_new_tokens"]
                            else "stop"
                        ),
                        response_id=None,
                    )
                )
        return records


def _merge_roles(messages: list[dict]) -> list[dict]:
    """Normalise a conversation for strict chat templates (e.g. Mistral): keep an
    optional leading system message, ensure the first message after it is a user
    turn (drop leading assistant turns introduced by context windowing), and
    collapse consecutive same-role messages by concatenating their contents."""
    out: list[dict] = []
    i = 0
    if messages and messages[0]["role"] == "system":
        out.append(dict(messages[0]))
        i = 1
    while i < len(messages) and messages[i]["role"] != "user":
        i += 1  # must start with a user turn after the system message
    for m in messages[i:]:
        if out and out[-1]["role"] == m["role"] and m["role"] != "system":
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _load_hf_model(
    model_id: str,
    device: str,
    dtype: str,
    quantization: str | None = None,
    *,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
):
    key = (model_id, model_revision, tokenizer_revision, device, dtype, quantization)
    if key in _HF_MODELS:
        return _HF_MODELS[key]
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from crucible.utils.seeding import seed_torch

    # Only one local model fits in GPU memory at a time; evict any previously
    # loaded model before loading a new one (the harness runs models serially).
    if _HF_MODELS:
        _HF_MODELS.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    seed_torch(0)
    _log.info("loading %s (%s, quant=%s) on %s ...", model_id, dtype, quantization, device)
    tok = AutoTokenizer.from_pretrained(
        model_id,
        revision=tokenizer_revision or model_revision,
        padding_side="left",
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs: dict = {"device_map": device}
    if quantization in {"4bit", "bitsandbytes-nf4"}:
        # NF4 double-quant; verified on Blackwell (sm_120). Lets ~32B fit in 32 GB.
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        dtype=getattr(torch, dtype),
        **kwargs,
    )
    model.eval()
    _HF_MODELS[key] = (tok, model)
    return tok, model


# ---------------------------------------------------------------------------
# API backends (dormant until credentials are provided)
# ---------------------------------------------------------------------------

_TRANSIENT_EXC = {
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "OverloadedError",
    "ServiceUnavailableError",
}


def _retry(fn: Callable[[], Any], *, retries: int = 8, base: float = 2.0, cap: float = 60.0) -> Any:
    """Call ``fn`` with exponential backoff on transient API errors (e.g. 429)."""
    import random
    import time

    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified below
            status = getattr(exc, "status_code", None)
            transient = type(exc).__name__ in _TRANSIENT_EXC or status in (429, 500, 502, 503, 529)
            if not transient or attempt == retries:
                raise
            delay = min(cap, base * (2**attempt)) + random.uniform(0, 1)
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    delay = max(delay, float(resp.headers.get("retry-after")))
                except (TypeError, ValueError, AttributeError):
                    pass
            _log.warning(
                "transient API error %s; retry %d/%d in %.1fs",
                type(exc).__name__,
                attempt + 1,
                retries,
                delay,
            )
            time.sleep(delay)


class AnthropicBackend(LLMBackend):
    """Claude API backend. Caches the static system prompt across calls."""

    def _generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]:
        import anthropic

        client = anthropic.Anthropic()

        def one(conv: Conversation) -> str:
            kwargs = dict(
                model=self.model_id,
                max_tokens=self.gen_params["max_new_tokens"],
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=conv,
            )
            temp = self.gen_params.get("temperature", 0.0)
            if temp is not None:
                kwargs["temperature"] = temp
            try:
                resp = _retry(lambda: client.messages.create(**kwargs))
            except anthropic.BadRequestError as exc:
                # Some models (e.g. Opus 4.8) deprecate temperature — drop and retry.
                if "temperature" in str(exc) and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    resp = _retry(lambda: client.messages.create(**kwargs))
                else:
                    raise
            return "".join(b.text for b in resp.content if b.type == "text")

        workers = int(self.gen_params.get("concurrency", 4))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, conversations))


class OpenAIBackend(LLMBackend):
    """OpenAI or any OpenAI-compatible server (set base_url for a local vLLM)."""

    def __init__(
        self,
        model_id: str,
        cache=None,
        base_url: str | None = None,
        model_revision: str | None = None,
        **gen_params,
    ):
        super().__init__(
            model_id,
            cache,
            base_url=base_url,
            model_revision=model_revision,
            **gen_params,
        )
        self._base_url = base_url
        self._model_revision = model_revision

    def _generate_batch(self, system: str, conversations: list[Conversation]) -> list[str]:
        return [record.response for record in self._generate_batch_records(system, conversations)]

    def _generate_batch_records(
        self,
        system: str,
        conversations: list[Conversation],
    ) -> list[GenerationRecord]:
        import openai

        # A local OpenAI-compatible server does not use a paid OpenAI credential.
        # Official OpenAI calls remain dormant and use the SDK's normal env lookup.
        client_kwargs: dict[str, Any] = {"base_url": self._base_url}
        if self._base_url:
            client_kwargs["api_key"] = "local-vllm"
        client = openai.OpenAI(**client_kwargs)
        # Hosted reasoning models reject temperature, while local
        # OpenAI-compatible servers such as vLLM accept it and need the explicit
        # value for genuinely greedy decoding. Reasoning effort is sent only
        # when explicitly configured.
        reasoning = bool(
            re.search(r"(^|/)(gpt-5|gpt-oss|o1|o3|o4)", self.model_id)
            or self.gen_params.get("reasoning_effort") is not None
        )

        def one(conv: Conversation) -> GenerationRecord:
            explicit_budget = self.gen_params.get("max_completion_tokens")
            max_completion_tokens = (
                int(explicit_budget)
                if explicit_budget is not None
                else max(self.gen_params["max_new_tokens"], 3000)
                if reasoning
                else self.gen_params["max_new_tokens"]
            )
            kwargs: dict = {
                "model": self.model_id,
                "messages": [{"role": "system", "content": system}, *conv],
                "max_completion_tokens": max_completion_tokens,
            }
            configured_effort = self.gen_params.get("reasoning_effort")
            if reasoning and configured_effort not in (None, "default"):
                kwargs["reasoning_effort"] = configured_effort
            if not reasoning or self._base_url:
                kwargs["temperature"] = self.gen_params.get("temperature", 0.0)
            if self.gen_params.get("seed") is not None:
                kwargs["seed"] = int(self.gen_params["seed"])
            if self.gen_params.get("extra_body"):
                kwargs["extra_body"] = self.gen_params["extra_body"]
            resp = _retry(lambda: client.chat.completions.create(**kwargs))
            choice = resp.choices[0]
            usage = (
                resp.usage.model_dump(mode="json")
                if getattr(resp, "usage", None) is not None
                else {}
            )
            effective = {
                "model": self.model_id,
                "model_revision": self._model_revision,
                "base_url": self._base_url,
                "max_completion_tokens": max_completion_tokens,
                "reasoning_effort": kwargs.get("reasoning_effort"),
                "temperature": kwargs.get("temperature"),
                "seed": kwargs.get("seed"),
                "extra_body": kwargs.get("extra_body"),
            }
            return GenerationRecord(
                response=choice.message.content or "",
                model_id=self.model_id,
                backend=self.backend_name,
                requested_params=dict(self.gen_params),
                effective_params=effective,
                usage=usage,
                finish_reason=getattr(choice, "finish_reason", None),
                response_id=getattr(resp, "id", None),
            )

        workers = int(self.gen_params.get("concurrency", 4))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, conversations))


_BACKENDS = {
    "mock": MockBackend,
    "transformers": TransformersBackend,
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
}


def make_backend(backend: str, model_id: str, **kwargs) -> LLMBackend:
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {list(_BACKENDS)}")
    return _BACKENDS[backend](model_id, **kwargs)
