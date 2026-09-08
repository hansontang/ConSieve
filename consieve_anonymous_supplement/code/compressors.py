"""Common sentence-compressor records and deterministic budget selection for M4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class ScoredSentence:
    document_index: int
    sentence_index: int
    text: str
    score: float


@dataclass
class CompressionOutput:
    selected_context: str
    selected_documents: list[str]
    selected_sentence_ids: list[tuple[int, int]]
    sentence_scores: list[ScoredSentence]
    retained_tokens: int
    original_tokens: int
    compression_latency_seconds: float = 0.0
    empty_context: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compression_ratio"] = (
            self.retained_tokens / self.original_tokens if self.original_tokens else 0.0
        )
        return payload


@dataclass(frozen=True)
class CompressionConfig:
    mode: str
    threshold: float | None = None
    token_budget: int | None = None
    threshold_inclusive: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class Compressor(Protocol):
    def compress(
        self, query: str, retrieved_documents: Sequence[str], config: CompressionConfig
    ) -> CompressionOutput: ...


class SentenceScoreCompressor:
    """Uniform adapter for any callable that returns document-indexed sentences."""

    def __init__(
        self,
        scorer: Callable[[str, Sequence[str]], Sequence[ScoredSentence]],
        token_count: Callable[[str], int],
    ) -> None:
        self.scorer = scorer
        self.token_count = token_count

    def compress(
        self, query: str, retrieved_documents: Sequence[str], config: CompressionConfig
    ) -> CompressionOutput:
        scores = list(self.scorer(query, retrieved_documents))
        if config.mode == "natural":
            if config.threshold is None:
                raise ValueError("natural mode requires threshold")
            return select_by_threshold(
                scores, len(retrieved_documents), config.threshold, self.token_count,
                retrieved_documents, inclusive=config.threshold_inclusive,
                metadata=config.metadata,
            )
        if config.mode == "hard_cap":
            if config.token_budget is None:
                raise ValueError("hard_cap mode requires token_budget")
            return select_by_hard_budget(
                scores, len(retrieved_documents), config.token_budget, self.token_count,
                retrieved_documents, metadata=config.metadata,
            )
        raise ValueError(f"unknown compression mode: {config.mode}")


def format_documents(
    documents: Sequence[str],
    context_format: str = "Retrieved Document: {content}",
    separator: str = "\n\n",
) -> str:
    return separator.join(
        context_format.format(rank=index + 1, content=text)
        for index, text in enumerate(documents)
        if text.strip()
    )


def _assemble(
    sentence_scores: Sequence[ScoredSentence], selected: set[tuple[int, int]], n_docs: int
) -> list[str]:
    by_document: list[list[str]] = [[] for _ in range(n_docs)]
    for item in sentence_scores:
        if (item.document_index, item.sentence_index) in selected:
            by_document[item.document_index].append(item.text.strip())
    return [" ".join(parts) for parts in by_document]


def select_by_threshold(
    sentence_scores: Sequence[ScoredSentence],
    n_docs: int,
    threshold: float,
    token_count: Callable[[str], int],
    original_documents: Sequence[str],
    inclusive: bool = False,
    metadata: dict[str, Any] | None = None,
) -> CompressionOutput:
    predicate = (lambda value: value >= threshold) if inclusive else (lambda value: value > threshold)
    selected = {
        (item.document_index, item.sentence_index)
        for item in sentence_scores
        if predicate(item.score)
    }
    documents = _assemble(sentence_scores, selected, n_docs)
    context = format_documents(documents)
    original_context = format_documents(original_documents)
    return CompressionOutput(
        selected_context=context,
        selected_documents=documents,
        selected_sentence_ids=sorted(selected),
        sentence_scores=list(sentence_scores),
        retained_tokens=token_count(context),
        original_tokens=token_count(original_context),
        empty_context=not bool(context.strip()),
        metadata={"selection": "natural_threshold", "threshold": threshold, **(metadata or {})},
    )


def select_by_hard_budget(
    sentence_scores: Sequence[ScoredSentence],
    n_docs: int,
    budget: int,
    token_count: Callable[[str], int],
    original_documents: Sequence[str],
    metadata: dict[str, Any] | None = None,
) -> CompressionOutput:
    """Greedily rank by score and enforce an exact final-context token cap.

    Ranking ties are resolved by original document/sentence order. Reconstruction
    always preserves that original order. A candidate is accepted only if the
    fully formatted context remains within the cap, so prefix/separator overhead
    is counted and no test-time approximation can exceed the requested budget.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    ranked = sorted(
        sentence_scores,
        key=lambda item: (-item.score, item.document_index, item.sentence_index),
    )
    selected: set[tuple[int, int]] = set()
    # Conservative additive accounting avoids repeatedly tokenizing a growing
    # context. The exact formatted context is checked below before returning.
    estimated_tokens = 0
    selected_documents: set[int] = set()
    prefix_cost = token_count("Retrieved Document: ") + 2
    for item in ranked:
        key = (item.document_index, item.sentence_index)
        cost = token_count(item.text) + 2
        if item.document_index not in selected_documents:
            cost += prefix_cost
        if estimated_tokens + cost <= budget:
            selected.add(key)
            selected_documents.add(item.document_index)
            estimated_tokens += cost
    # Exact enforcement includes tokenizer boundary effects and formatting.
    # Remove the lowest-ranked selected sentence until the hard cap is met.
    selected_ranked = [
        (item.document_index, item.sentence_index)
        for item in ranked
        if (item.document_index, item.sentence_index) in selected
    ]
    while selected_ranked:
        exact_context = format_documents(_assemble(sentence_scores, selected, n_docs))
        if token_count(exact_context) <= budget:
            break
        selected.remove(selected_ranked.pop())
    documents = _assemble(sentence_scores, selected, n_docs)
    context = format_documents(documents)
    retained = token_count(context)
    if retained > budget:
        raise AssertionError(f"hard-cap violation: {retained} > {budget}")
    original_context = format_documents(original_documents)
    return CompressionOutput(
        selected_context=context,
        selected_documents=documents,
        selected_sentence_ids=sorted(selected),
        sentence_scores=list(sentence_scores),
        retained_tokens=retained,
        original_tokens=token_count(original_context),
        empty_context=not bool(context.strip()),
        metadata={"selection": "per_query_hard_cap", "budget": budget, **(metadata or {})},
    )
