"""
LLM-as-a-Judge evaluation module.

Uses an LLM to judge if model answers match reference answers.
"""

import re
import json
import hashlib
import sqlite3
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from .llm_client import create_llm_client, BatchGenerator, BaseLLMClient
from .metrics import normalize_answer, extract_answers, dedupe_answers


# Regex to extract Yes/No
YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
JUDGE_PROMPT_VERSION = "clean_v2_yes_no_v1"


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class JudgeCache:
    """Per-reference judge decisions for reproducible subgroup aggregation."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS judge_cache (
                query_id TEXT NOT NULL,
                prediction_hash TEXT NOT NULL,
                reference_hash TEXT NOT NULL,
                model_alias TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                params_hash TEXT NOT NULL,
                judgment TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                PRIMARY KEY (
                    query_id, prediction_hash, reference_hash, model_alias,
                    model_version, prompt_version, params_hash
                )
            )
            """
        )
        self.connection.commit()

    def get(self, key: tuple[str, ...]) -> str | None:
        row = self.connection.execute(
            """SELECT judgment FROM judge_cache WHERE
            query_id=? AND prediction_hash=? AND reference_hash=? AND model_alias=?
            AND model_version=? AND prompt_version=? AND params_hash=?""",
            key,
        ).fetchone()
        return row[0] if row else None

    def put(self, key: tuple[str, ...], judgment: str) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO judge_cache
            (query_id, prediction_hash, reference_hash, model_alias, model_version,
             prompt_version, params_hash, judgment, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (*key, judgment, dt.datetime.now(dt.timezone.utc).isoformat()),
        )
        self.connection.commit()


@dataclass
class JudgmentResult:
    """Result of a single judgment."""
    entry_id: str
    question: str
    prediction: str
    reference: str
    judgment: str  # "yes", "no", or "error"
    is_correct: bool


@dataclass
class JudgeMetrics:
    """Aggregated judge metrics."""
    total_entries: int = 0
    entries_with_refs: int = 0
    correct: int = 0
    accuracy: float = 0.0
    judgments: List[JudgmentResult] = None
    cache_hits: int = 0
    cache_misses: int = 0
    api_failure_count: int = 0
    retry_count: int = 0

    def __post_init__(self):
        if self.judgments is None:
            self.judgments = []


def build_judge_prompt(
    question: str,
    reference: str,
    prediction: str,
) -> List[Dict[str, str]]:
    """Build messages for judge LLM."""
    return [
        {
            "role": "system",
            "content": "You are a strict judge. Respond only with Yes or No."
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Reference Answer: {reference}\n"
                f"Model Answer: {prediction}\n"
                "Does the model answer match the reference answer? "
                "Reply only Yes or No."
            ),
        },
    ]


def extract_yes_no(text: str) -> str:
    """Extract yes/no from judge response."""
    if not text:
        return "error"
    match = YES_NO_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    return "error"


class LLMJudge:
    """
    LLM-as-a-Judge evaluator.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        batch_size: int = 128,
        sampling_params: Dict[str, Any] = None,
        cache_path: str | None = None,
        model_alias: str = "unknown",
        model_version: str | None = None,
        prompt_version: str = JUDGE_PROMPT_VERSION,
    ):
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.sampling_params = sampling_params or {
            "max_tokens": 8,
            "temperature": 0.0,
        }
        self.cache = JudgeCache(cache_path) if cache_path else None
        self.model_alias = model_alias
        self.model_version = model_version or "unavailable"
        self.prompt_version = prompt_version
        self.params_hash = _hash_value(
            json.dumps(self.sampling_params, sort_keys=True, separators=(",", ":"))
        )

    def _cache_key(self, entry_id: str, prediction: str, reference: str) -> tuple[str, ...]:
        return (
            entry_id,
            _hash_value(prediction),
            _hash_value(reference),
            self.model_alias,
            self.model_version,
            self.prompt_version,
            self.params_hash,
        )

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        prediction_field: str = "model_answer",
        question_field: str = "question",
        id_field: str = "id",
        answer_fields: List[str] = None,
    ) -> JudgeMetrics:
        """
        Evaluate predictions using LLM judge.

        For each entry, we check if the model answer matches ANY of the
        reference answers (one prompt per reference).

        Args:
            predictions: List of prediction entries
            prediction_field: Field name for model prediction
            question_field: Field name for question
            id_field: Field name for entry ID
            answer_fields: List of fields containing reference answers

        Returns:
            JudgeMetrics with accuracy and per-entry judgments
        """
        if answer_fields is None:
            answer_fields = ["answers", "additional_answers", "answers_additional"]

        # Prepare all judgment prompts
        prompts = []
        prompt_mapping = []  # (entry_idx, reference_answer, cache_key)

        entries_with_refs = []
        cached_judgments: list[tuple[int, str, str]] = []
        cache_hits = 0

        for idx, entry in enumerate(predictions):
            prediction = entry.get(prediction_field, "")
            question = entry.get(question_field, "")
            references = dedupe_answers(extract_answers(entry, answer_fields))

            if not references:
                continue

            entries_with_refs.append(idx)

            for ref in references:
                entry_id = str(entry.get(id_field, ""))
                cache_key = self._cache_key(entry_id, prediction, ref)
                cached = self.cache.get(cache_key) if self.cache else None
                if cached is not None:
                    cached_judgments.append((idx, ref, cached))
                    cache_hits += 1
                    continue
                messages = build_judge_prompt(question, ref, prediction)
                prompts.append(messages)
                prompt_mapping.append((idx, ref, cache_key))

        if not prompts and not cached_judgments:
            return JudgeMetrics(total_entries=len(predictions))

        # Batch generate judgments
        responses = []
        if prompts:
            print(f"Running LLM judge on {len(prompts)} uncached judgments...")
            generator = BatchGenerator(
                client=self.llm_client,
                batch_size=self.batch_size,
                sampling_params=self.sampling_params,
            )
            responses = generator.generate_all(prompts, desc="Judging")

        # Process responses
        # For each entry, we mark it correct if ANY reference matches (yes)
        entry_correct = {idx: False for idx in entries_with_refs}
        entry_judgments = {idx: [] for idx in entries_with_refs}

        for entry_idx, ref, judgment in cached_judgments:
            entry = predictions[entry_idx]
            result = JudgmentResult(
                entry_id=entry.get(id_field, ""),
                question=entry.get(question_field, ""),
                prediction=entry.get(prediction_field, ""),
                reference=ref,
                judgment=judgment,
                is_correct=(judgment == "yes"),
            )
            entry_judgments[entry_idx].append(result)
            if judgment == "yes":
                entry_correct[entry_idx] = True

        api_failure_count = 0
        for (entry_idx, ref, cache_key), response in zip(prompt_mapping, responses):
            judgment = extract_yes_no(response.text)
            if judgment == "error":
                api_failure_count += 1
            # Provider/parse failures are not cached as decisions; a later
            # explicitly logged retry may judge them once successfully.
            if self.cache and judgment != "error":
                self.cache.put(cache_key, judgment)

            entry = predictions[entry_idx]
            result = JudgmentResult(
                entry_id=entry.get(id_field, ""),
                question=entry.get(question_field, ""),
                prediction=entry.get(prediction_field, ""),
                reference=ref,
                judgment=judgment,
                is_correct=(judgment == "yes"),
            )
            entry_judgments[entry_idx].append(result)

            if judgment == "yes":
                entry_correct[entry_idx] = True

        # Aggregate metrics
        correct_count = sum(1 for v in entry_correct.values() if v)
        total_with_refs = len(entries_with_refs)

        metrics = JudgeMetrics(
            total_entries=len(predictions),
            entries_with_refs=total_with_refs,
            correct=correct_count,
            accuracy=correct_count / total_with_refs if total_with_refs > 0 else 0.0,
            judgments=[j for judgments in entry_judgments.values() for j in judgments],
            cache_hits=cache_hits,
            cache_misses=len(prompts),
            api_failure_count=api_failure_count,
            retry_count=0,
        )

        return metrics


def aggregate_cached_judgments(
    predictions: List[Dict[str, Any]],
    judgments: List[JudgmentResult],
    id_field: str = "id",
) -> JudgeMetrics:
    """Aggregate an already-judged superset without any provider call."""
    entry_ids = [str(entry.get(id_field, "")) for entry in predictions]
    wanted = set(entry_ids)
    selected = [judgment for judgment in judgments if str(judgment.entry_id) in wanted]
    by_id: Dict[str, List[JudgmentResult]] = {}
    for judgment in selected:
        by_id.setdefault(str(judgment.entry_id), []).append(judgment)
    correct = sum(1 for entry_id in entry_ids if any(j.is_correct for j in by_id.get(entry_id, [])))
    n_judge = sum(1 for entry_id in entry_ids if by_id.get(entry_id))
    return JudgeMetrics(
        total_entries=len(predictions),
        entries_with_refs=n_judge,
        correct=correct,
        accuracy=correct / n_judge if n_judge else 0.0,
        judgments=selected,
    )


def run_llm_judge(
    predictions_path: str,
    config: Dict[str, Any],
    output_path: str = None,
) -> JudgeMetrics:
    """
    Run LLM judge on a predictions file.

    Args:
        predictions_path: Path to predictions JSON file
        config: Configuration dictionary with 'judge' section
        output_path: Optional path to save judgment results

    Returns:
        JudgeMetrics
    """
    # Load predictions
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions from {predictions_path}")

    # Create judge client
    judge_cfg = config.get("judge", {})
    llm_client = create_llm_client(judge_cfg)

    # Create judge
    judge = LLMJudge(
        llm_client=llm_client,
        batch_size=judge_cfg.get("batch_size", 128),
        sampling_params=judge_cfg.get("sampling", {}),
    )

    # Run evaluation
    metrics = judge.evaluate(predictions)

    # Print results
    print("\n" + "=" * 60)
    print("LLM Judge Results")
    print("=" * 60)
    print(f"Total entries:        {metrics.total_entries}")
    print(f"Entries with refs:    {metrics.entries_with_refs}")
    print(f"Correct:              {metrics.correct}")
    print(f"Accuracy:             {metrics.accuracy:.4f}")
    print("=" * 60)

    # Save results
    if output_path:
        results = {
            "total_entries": metrics.total_entries,
            "entries_with_refs": metrics.entries_with_refs,
            "correct": metrics.correct,
            "accuracy": metrics.accuracy,
            "judgments": [
                {
                    "entry_id": j.entry_id,
                    "question": j.question,
                    "prediction": j.prediction,
                    "reference": j.reference,
                    "judgment": j.judgment,
                    "is_correct": j.is_correct,
                }
                for j in metrics.judgments
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_path}")

    return metrics


def format_judge_report(metrics: JudgeMetrics) -> str:
    """Format judge metrics as a report string."""
    lines = [
        "=" * 60,
        "LLM Judge Results",
        "=" * 60,
        f"Total entries:        {metrics.total_entries}",
        f"Entries with refs:    {metrics.entries_with_refs}",
        f"Correct:              {metrics.correct}",
        f"Accuracy:             {metrics.accuracy:.4f}",
        "=" * 60,
    ]
    return "\n".join(lines)
