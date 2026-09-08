"""Pinned, method-independent context token accounting for clean_v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


DEFAULT_TOKENIZER_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TOKENIZER_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


class Encoder(Protocol):
    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


@dataclass(frozen=True)
class TokenizerSpec:
    name: str = DEFAULT_TOKENIZER_NAME
    revision: str = DEFAULT_TOKENIZER_REVISION


class ContextTokenCounter:
    """Count context-only tokens with one pinned Hugging Face tokenizer."""

    def __init__(self, spec: TokenizerSpec | None = None, tokenizer: Encoder | None = None):
        self.spec = spec or TokenizerSpec()
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Encoder:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.spec.name,
                revision=self.spec.revision,
                trust_remote_code=False,
            )
        return self._tokenizer

    def count(self, context: str) -> int:
        if not context:
            return 0
        return len(self.tokenizer.encode(context, add_special_tokens=False))


def create_context_token_counter(config: dict[str, Any]) -> ContextTokenCounter | None:
    token_cfg = config.get("token_accounting")
    if not token_cfg or not token_cfg.get("enabled", False):
        return None
    spec = TokenizerSpec(
        name=token_cfg.get("tokenizer_name", DEFAULT_TOKENIZER_NAME),
        revision=token_cfg.get("tokenizer_revision", DEFAULT_TOKENIZER_REVISION),
    )
    return ContextTokenCounter(spec=spec)

