"""
Evaluation metrics: F1, Exact Match, and token counting.
"""

import re
import string
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# Punctuation translation table
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Articles to remove
_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t and t not in _ARTICLES]
    return " ".join(tokens)


def compute_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score between prediction and reference.
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Count token occurrences
    pred_counts = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1

    ref_counts = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    # Compute overlap
    overlap = 0
    for tok, count in pred_counts.items():
        overlap += min(count, ref_counts.get(tok, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_exact_match(prediction: str, reference: str) -> float:
    """
    Compute exact match score (1.0 if normalized strings match, 0.0 otherwise).
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def extract_answers(entry: Dict[str, Any], answer_fields: List[str] = None) -> List[str]:
    """
    Extract all reference answers from an entry.
    """
    if answer_fields is None:
        answer_fields = ["answers", "additional_answers", "answers_additional"]

    answers = []
    for field in answer_fields:
        value = entry.get(field)
        if isinstance(value, list):
            answers.extend([a for a in value if isinstance(a, str)])
        elif isinstance(value, str):
            answers.append(value)
    return answers


def dedupe_answers(answers: List[str]) -> List[str]:
    """
    Deduplicate answers by normalized form while preserving original text.
    """
    seen = set()
    unique = []
    for ans in answers:
        norm = normalize_answer(ans)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(ans)
    return unique


@dataclass
class MetricResult:
    """Result for a single entry."""
    id: str
    prediction: str
    references: List[str]
    f1: float
    em: float
    best_reference: str = ""


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all entries."""
    total_entries: int = 0
    entries_with_refs: int = 0
    n_total: int = 0
    n_qa: int = 0
    n_token: int = 0
    n_judge: int = 0

    # F1 metrics
    f1_sum: float = 0.0
    f1_micro: float = 0.0  # Average F1

    # Exact Match metrics
    em_sum: float = 0.0
    em_micro: float = 0.0  # Average EM (accuracy)

    # LLM Judge metrics (optional, set after running judge)
    judge_correct: int = 0
    judge_accuracy: float = 0.0

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_context_tokens: int = 0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_context_tokens: float = 0.0

    # GT retrieval rate
    gt_retrieved_count: int = 0
    gt_total_count: int = 0
    gt_retrieval_rate: float = 0.0

    # Per-entry results
    results: List[MetricResult] = field(default_factory=list)


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    prediction_field: str = "model_answer",
    id_field: str = "id",
    answer_fields: List[str] = None,
    input_token_field: str = "input_tokens",
    output_token_field: str = "output_tokens",
    context_token_field: str = "context_tokens",
) -> AggregatedMetrics:
    """
    Evaluate a list of predictions against reference answers.

    Args:
        predictions: List of prediction entries with answers and model outputs
        prediction_field: Field name for model prediction
        id_field: Field name for entry ID
        answer_fields: List of fields containing reference answers
        input_token_field: Field name for input token count
        output_token_field: Field name for output token count

    Returns:
        AggregatedMetrics with F1, EM, and token usage
    """
    if answer_fields is None:
        answer_fields = ["answers", "additional_answers", "answers_additional"]

    metrics = AggregatedMetrics()

    for entry in predictions:
        metrics.total_entries += 1
        metrics.n_total += 1

        prediction = entry.get(prediction_field, "")
        references = dedupe_answers(extract_answers(entry, answer_fields))

        # Token counting
        input_tokens = entry.get(input_token_field)
        output_tokens = entry.get(output_token_field)
        context_tokens = entry.get(context_token_field)
        has_token_measurement = any(
            value is not None for value in (input_tokens, output_tokens, context_tokens)
        )
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        context_tokens = context_tokens or 0
        metrics.total_input_tokens += input_tokens
        metrics.total_output_tokens += output_tokens
        metrics.total_context_tokens += context_tokens
        if has_token_measurement:
            metrics.n_token += 1

        # GT retrieval tracking
        gt_retrieved = entry.get("gt_retrieved", None)
        if gt_retrieved is not None:
            metrics.gt_total_count += 1
            if gt_retrieved:
                metrics.gt_retrieved_count += 1

        if not references:
            continue

        metrics.entries_with_refs += 1
        metrics.n_qa += 1

        # Compute best F1 and EM across all references
        best_f1 = 0.0
        best_em = 0.0
        best_ref = ""

        for ref in references:
            f1 = compute_f1(prediction, ref)
            em = compute_exact_match(prediction, ref)

            if f1 > best_f1:
                best_f1 = f1
                best_ref = ref

            if em > best_em:
                best_em = em

        metrics.f1_sum += best_f1
        metrics.em_sum += best_em

        result = MetricResult(
            id=entry.get(id_field, ""),
            prediction=prediction,
            references=references,
            f1=best_f1,
            em=best_em,
            best_reference=best_ref,
        )
        metrics.results.append(result)

    # Compute averages
    if metrics.entries_with_refs > 0:
        metrics.f1_micro = metrics.f1_sum / metrics.entries_with_refs
        metrics.em_micro = metrics.em_sum / metrics.entries_with_refs

    if metrics.n_token > 0:
        metrics.avg_input_tokens = metrics.total_input_tokens / metrics.n_token
        metrics.avg_output_tokens = metrics.total_output_tokens / metrics.n_token
        metrics.avg_context_tokens = metrics.total_context_tokens / metrics.n_token

    if metrics.gt_total_count > 0:
        metrics.gt_retrieval_rate = metrics.gt_retrieved_count / metrics.gt_total_count

    return metrics


def count_tokens(tokenizer, text: str) -> Optional[int]:
    """
    Count tokens in text using a tokenizer.
    """
    if tokenizer is None:
        return None

    try:
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text))
    except Exception:
        pass

    try:
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return len(encoded["input_ids"])
        if hasattr(encoded, "input_ids"):
            return len(encoded.input_ids)
    except Exception:
        pass

    return None


@dataclass
class GroupedMetrics:
    """Metrics grouped by perturbation status and dataset source."""
    # Main categories
    all: AggregatedMetrics = None
    non_perturbed: AggregatedMetrics = None
    perturbed: AggregatedMetrics = None

    # Dataset source categories
    streamingqa: AggregatedMetrics = None
    timeqa: AggregatedMetrics = None

    # Cross categories (source x perturbation)
    streamingqa_non_perturbed: AggregatedMetrics = None
    streamingqa_perturbed: AggregatedMetrics = None
    timeqa_non_perturbed: AggregatedMetrics = None
    timeqa_perturbed: AggregatedMetrics = None

    def __post_init__(self):
        if self.all is None:
            self.all = AggregatedMetrics()
        if self.non_perturbed is None:
            self.non_perturbed = AggregatedMetrics()
        if self.perturbed is None:
            self.perturbed = AggregatedMetrics()
        if self.streamingqa is None:
            self.streamingqa = AggregatedMetrics()
        if self.timeqa is None:
            self.timeqa = AggregatedMetrics()
        if self.streamingqa_non_perturbed is None:
            self.streamingqa_non_perturbed = AggregatedMetrics()
        if self.streamingqa_perturbed is None:
            self.streamingqa_perturbed = AggregatedMetrics()
        if self.timeqa_non_perturbed is None:
            self.timeqa_non_perturbed = AggregatedMetrics()
        if self.timeqa_perturbed is None:
            self.timeqa_perturbed = AggregatedMetrics()


def evaluate_predictions_grouped(
    predictions: List[Dict[str, Any]],
    prediction_field: str = "model_answer",
    id_field: str = "id",
    answer_fields: List[str] = None,
    input_token_field: str = "input_tokens",
    output_token_field: str = "output_tokens",
    perturbation_field: str = "is_hard_negative",
    source_field: str = "source",
) -> GroupedMetrics:
    """
    Evaluate predictions with breakdown by perturbation status and dataset source.

    Returns metrics for 9 categories:
    - all: All samples
    - non_perturbed: Samples where is_hard_negative=False
    - perturbed: Samples where is_hard_negative=True
    - streamingqa: All StreamingQA samples
    - timeqa: All TimeQA samples
    - streamingqa_non_perturbed: StreamingQA non-perturbed
    - streamingqa_perturbed: StreamingQA perturbed
    - timeqa_non_perturbed: TimeQA non-perturbed
    - timeqa_perturbed: TimeQA perturbed
    """
    if answer_fields is None:
        answer_fields = ["answers", "additional_answers", "answers_additional"]

    def is_streamingqa(p):
        source = p.get(source_field, "")
        return source.startswith("streamingqa")

    def is_timeqa(p):
        source = p.get(source_field, "")
        return source.startswith("timeqa")

    def is_perturbed(p):
        return p.get(perturbation_field, False)

    # Split predictions by perturbation status
    all_preds = predictions
    non_perturbed_preds = [p for p in predictions if not is_perturbed(p)]
    perturbed_preds = [p for p in predictions if is_perturbed(p)]

    # Split by source
    streamingqa_preds = [p for p in predictions if is_streamingqa(p)]
    timeqa_preds = [p for p in predictions if is_timeqa(p)]

    # Cross categories
    streamingqa_non_perturbed_preds = [p for p in streamingqa_preds if not is_perturbed(p)]
    streamingqa_perturbed_preds = [p for p in streamingqa_preds if is_perturbed(p)]
    timeqa_non_perturbed_preds = [p for p in timeqa_preds if not is_perturbed(p)]
    timeqa_perturbed_preds = [p for p in timeqa_preds if is_perturbed(p)]

    def eval_preds(preds):
        return evaluate_predictions(
            preds, prediction_field, id_field, answer_fields,
            input_token_field, output_token_field
        )

    return GroupedMetrics(
        all=eval_preds(all_preds),
        non_perturbed=eval_preds(non_perturbed_preds),
        perturbed=eval_preds(perturbed_preds),
        streamingqa=eval_preds(streamingqa_preds),
        timeqa=eval_preds(timeqa_preds),
        streamingqa_non_perturbed=eval_preds(streamingqa_non_perturbed_preds),
        streamingqa_perturbed=eval_preds(streamingqa_perturbed_preds),
        timeqa_non_perturbed=eval_preds(timeqa_non_perturbed_preds),
        timeqa_perturbed=eval_preds(timeqa_perturbed_preds),
    )


def format_metrics_report(metrics: AggregatedMetrics, method_name: str = "") -> str:
    """
    Format metrics as a human-readable report.
    """
    lines = []
    lines.append("=" * 60)
    if method_name:
        lines.append(f"Evaluation Results: {method_name}")
    else:
        lines.append("Evaluation Results")
    lines.append("=" * 60)
    lines.append(f"Total entries:        {metrics.total_entries}")
    lines.append(f"Entries with refs:    {metrics.entries_with_refs}")
    lines.append(
        f"Denominators:         total={metrics.n_total}, qa={metrics.n_qa}, "
        f"token={metrics.n_token}, judge={metrics.n_judge}"
    )
    lines.append("")
    lines.append("Metrics:")
    lines.append(f"  F1 (micro):         {metrics.f1_micro:.4f}")
    lines.append(f"  Exact Match:        {metrics.em_micro:.4f}")
    if metrics.gt_total_count > 0:
        lines.append(f"  GT Retrieval Rate:  {metrics.gt_retrieval_rate:.4f} ({metrics.gt_retrieved_count}/{metrics.gt_total_count})")
    lines.append("")
    lines.append("Token Usage:")
    lines.append(f"  Total input:        {metrics.total_input_tokens:,}")
    lines.append(f"  Total output:       {metrics.total_output_tokens:,}")
    lines.append(f"  Avg input/query:    {metrics.avg_input_tokens:.1f}")
    lines.append(f"  Avg context/query:  {metrics.avg_context_tokens:.1f}")
    lines.append(f"  Avg output/query:   {metrics.avg_output_tokens:.1f}")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_grouped_metrics_report(grouped: GroupedMetrics, method_name: str = "") -> str:
    """
    Format grouped metrics as a human-readable report with breakdown.
    """
    # Check if judge metrics are available
    has_judge = grouped.all.judge_accuracy > 0 or grouped.all.judge_correct > 0
    # Check if GT retrieval metrics are available
    has_gt_ret = grouped.all.gt_total_count > 0

    lines = []
    width = 90
    if has_judge:
        width += 10
    if has_gt_ret:
        width += 10
    lines.append("=" * width)
    if method_name:
        lines.append(f"Evaluation Results: {method_name}")
    else:
        lines.append("Evaluation Results")
    lines.append("=" * width)

    # Summary table header
    lines.append("")
    header = f"{'Category':<28} {'nT/nQ/nTok/nJ':>18} {'F1':>10} {'EM':>10}"
    if has_judge:
        header += f" {'Judge':>10}"
    if has_gt_ret:
        header += f" {'GT Ret':>10}"
    header += f" {'Avg Ctx':>10} {'Avg In':>10}"
    lines.append(header)
    lines.append("-" * width)

    # All 9 categories
    categories = [
        ("All", grouped.all),
        ("Non-Perturbed", grouped.non_perturbed),
        ("Perturbed", grouped.perturbed),
        ("StreamingQA", grouped.streamingqa),
        ("StreamingQA Non-Perturbed", grouped.streamingqa_non_perturbed),
        ("StreamingQA Perturbed", grouped.streamingqa_perturbed),
        ("TimeQA", grouped.timeqa),
        ("TimeQA Non-Perturbed", grouped.timeqa_non_perturbed),
        ("TimeQA Perturbed", grouped.timeqa_perturbed),
    ]

    for name, m in categories:
        if m.entries_with_refs > 0:
            denominators = f"{m.n_total}/{m.n_qa}/{m.n_token}/{m.n_judge}"
            row = f"{name:<28} {denominators:>18} {m.f1_micro:>10.4f} {m.em_micro:>10.4f}"
            if has_judge:
                row += f" {m.judge_accuracy:>10.4f}"
            if has_gt_ret:
                if m.gt_total_count > 0:
                    row += f" {m.gt_retrieval_rate:>10.4f}"
                else:
                    row += f" {'N/A':>10}"
            row += f" {m.avg_context_tokens:>10.1f} {m.avg_input_tokens:>10.1f}"
            lines.append(row)
        else:
            row = f"{name:<28} {'N/A':>18} {'N/A':>10} {'N/A':>10}"
            if has_judge:
                row += f" {'N/A':>10}"
            if has_gt_ret:
                row += f" {'N/A':>10}"
            row += f" {'N/A':>10} {'N/A':>10}"
            lines.append(row)

    lines.append("-" * width)

    # Token totals
    lines.append("")
    lines.append("Token Usage (All):")
    lines.append(f"  Total context:      {grouped.all.total_context_tokens:,}")
    lines.append(f"  Total input:        {grouped.all.total_input_tokens:,}")
    lines.append(f"  Total output:       {grouped.all.total_output_tokens:,}")

    lines.append("=" * width)
    return "\n".join(lines)
