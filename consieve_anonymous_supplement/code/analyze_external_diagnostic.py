#!/usr/bin/env python3
"""Aggregate external-baseline stress scores with the frozen M2 metric definitions."""

from __future__ import annotations

import argparse
import csv
import json
from statistics import mean, median
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_m2_stress import auroc, average_precision, groups, read_jsonl


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def pair_metrics(row: dict) -> dict[str, float | None]:
    """Use frozen per-sentence decisions so each method keeps its own comparator.

    The four M2 artifacts encode strict-threshold decisions; the official EXIT
    artifact encodes its inclusive >=0.5 decision. Scores remain untouched for
    gap, pair discrimination, and ranking metrics.
    """
    indices = [i - 1 for i in row["positive_evidence_indices"] if 1 <= i <= len(row["positive_scores"])]
    positive = [float(row["positive_scores"][i]) for i in indices]
    negative = [float(row["perturbed_scores"][i]) for i in indices]
    positive_retained = [bool(row["positive_retained"][i]) for i in indices]
    negative_retained = [bool(row["perturbed_retained"][i]) for i in indices]
    positive_recall = safe_mean([float(value) for value in positive_retained])
    invalid_retention = safe_mean([float(value) for value in negative_retained])
    gaps = [a - b for a, b in zip(positive, negative)]
    return {
        "positive_evidence_recall": positive_recall,
        "invalid_evidence_retention": invalid_retention,
        "constraint_rejection_rate": None if invalid_retention is None else 1 - invalid_retention,
        "score_gap_mean": safe_mean(gaps),
        "score_gap_median": median(gaps) if gaps else None,
        "positive_gap_rate": safe_mean([float(value > 0) for value in gaps]),
        "pairwise_discrimination_accuracy": safe_mean([float(a > b) for a, b in zip(positive, negative)]),
        "positive_empty_rate": float(row["positive_empty"]),
        "perturbed_empty_rate": float(row["perturbed_empty"]),
        "constraint_sensitivity_delta": None if positive_recall is None or invalid_retention is None else positive_recall - invalid_retention,
    }


def aggregate(rows: list[dict]) -> dict:
    paired = [pair_metrics(row) for row in rows]
    evidence_positive, evidence_negative = [], []
    for row in rows:
        for index in row["positive_evidence_indices"]:
            if 1 <= index <= len(row["positive_scores"]):
                evidence_positive.append(float(row["positive_scores"][index - 1]))
                evidence_negative.append(float(row["perturbed_scores"][index - 1]))
    labels = [1] * len(evidence_positive) + [0] * len(evidence_negative)
    scores = evidence_positive + evidence_negative
    result = {"n_pairs": len(rows), "n_evidence_sentences": len(evidence_positive)}
    for key in (
        "positive_evidence_recall", "invalid_evidence_retention", "constraint_rejection_rate",
        "score_gap_mean", "score_gap_median", "positive_gap_rate",
        "pairwise_discrimination_accuracy", "positive_empty_rate", "perturbed_empty_rate",
        "constraint_sensitivity_delta",
    ):
        result[key] = safe_mean([float(item[key]) for item in paired if item[key] is not None])
    result["auroc"] = auroc(labels, scores)
    result["auprc"] = average_precision(labels, scores)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        records = read_jsonl(path)
        method = records[0]["model_key"]
        for group_type, group, subset in groups(records):
            row = aggregate(subset)
            for field in ("positive_sentence_alignment_rate", "negative_sentence_alignment_rate"):
                values = [float(item[field]) for item in subset if item.get(field) is not None]
                row[field] = sum(values) / len(values) if values else None
            row["scores_exactly_at_threshold"] = sum(
                score == float(item["threshold"])
                for item in subset
                for score in item["positive_scores"] + item["perturbed_scores"]
            )
            row.update({"method": method, "group_type": group_type, "group": group, "raw_output": str(path)})
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
