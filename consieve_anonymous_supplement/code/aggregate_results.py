#!/usr/bin/env python3
"""Aggregate raw Clean Evaluation v2 outputs without a Google Sheet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import compute_exact_match, compute_f1, dedupe_answers, extract_answers


def load_records(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Raw output must contain a list: {path}")
    return data


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    f1_values = []
    em_values = []
    context_values = []
    input_values = []
    judge_values = []
    content_filtered = 0
    compression_ratios = []
    compression_latencies = []
    empty_contexts = 0
    evidence_tp = evidence_fp = evidence_fn = 0
    evidence_records = 0
    retrieval_values = []
    provider_errors = 0
    judge_failures = 0
    for record in records:
        references = dedupe_answers(extract_answers(record))
        if references:
            prediction = record.get("model_answer", "")
            f1_values.append(max(compute_f1(prediction, reference) for reference in references))
            em_values.append(max(compute_exact_match(prediction, reference) for reference in references))
        if record.get("context_tokens") is not None:
            context_values.append(int(record["context_tokens"]))
        if record.get("input_tokens") is not None:
            input_values.append(int(record["input_tokens"]))
        if record.get("judge_status") == "judged":
            judge_values.append(bool(record.get("judge_correct")))
        if str(record.get("model_answer", "")).strip() == "[CONTENT_FILTERED]":
            content_filtered += 1
        if str(record.get("model_answer", "")).lstrip().startswith("[ERROR:"):
            provider_errors += 1
        if record.get("judge_status") not in (None, "judged"):
            judge_failures += 1
        if not str(record.get("context", "")).strip():
            empty_contexts += 1
        if record.get("compression_ratio") is not None:
            compression_ratios.append(float(record["compression_ratio"]))
        if record.get("compression_latency_seconds") is not None:
            compression_latencies.append(float(record["compression_latency_seconds"]))
        if all(record.get(key) is not None for key in ("evidence_tp", "evidence_fp", "evidence_fn")):
            evidence_records += 1
            evidence_tp += int(record["evidence_tp"])
            evidence_fp += int(record["evidence_fp"])
            evidence_fn += int(record["evidence_fn"])
        if record.get("gt_retrieved") is not None:
            retrieval_values.append(bool(record["gt_retrieved"]))

    return {
        "n_total": len(records),
        "n_qa": len(f1_values),
        "n_token": len(context_values),
        "n_judge": len(judge_values),
        "n_content_filtered": content_filtered,
        "n_provider_errors": provider_errors,
        "n_judge_failures": judge_failures,
        "n_evidence": evidence_records,
        "n_retrieval": len(retrieval_values),
        "f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "em": sum(em_values) / len(em_values) if em_values else None,
        "judge": sum(judge_values) / len(judge_values) if judge_values else None,
        "context_tokens": sum(context_values) / len(context_values) if context_values else None,
        "total_input_tokens": sum(input_values) / len(input_values) if input_values else None,
        "evidence_precision": (
            evidence_tp / (evidence_tp + evidence_fp)
            if evidence_tp + evidence_fp else (0.0 if evidence_records else None)
        ),
        "evidence_recall": (
            evidence_tp / (evidence_tp + evidence_fn)
            if evidence_tp + evidence_fn else (0.0 if evidence_records else None)
        ),
        "compression_ratio": (
            sum(compression_ratios) / len(compression_ratios) if compression_ratios else None
        ),
        "compression_latency_seconds": (
            sum(compression_latencies) / len(compression_latencies)
            if compression_latencies else None
        ),
        "empty_context_rate": empty_contexts / len(records) if records else None,
        "retrieval_recall": (
            sum(retrieval_values) / len(retrieval_values) if retrieval_values else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--allow-subset-denominators", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        records = load_records(path)
        subgroups = {
            "All": records,
            "StreamingQA": [r for r in records if str(r.get("source", "")).startswith("streamingqa")],
            "TimeQA": [r for r in records if str(r.get("source", "")).startswith("timeqa")],
        }
        for dataset, subset in subgroups.items():
            if not subset:
                continue
            metrics = aggregate(subset)
            metrics["method"] = path.stem
            metrics["dataset"] = dataset
            if not args.allow_subset_denominators:
                denominators = {
                    metrics["n_total"], metrics["n_qa"], metrics["n_token"], metrics["n_judge"]
                }
                if len(denominators) != 1:
                    raise ValueError(f"Denominator mismatch for {path} ({dataset}): {sorted(denominators)}")
            rows.append(metrics)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump({"protocol_version": "clean_v2", "rows": rows}, handle, indent=2)
        handle.write("\n")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method", "dataset", "n_total", "n_qa", "n_token", "n_judge",
            "n_evidence", "n_retrieval", "n_content_filtered", "n_provider_errors",
            "n_judge_failures", "f1", "em", "judge", "evidence_precision",
            "evidence_recall", "context_tokens", "total_input_tokens", "compression_ratio",
            "compression_latency_seconds", "empty_context_rate", "retrieval_recall",
        ])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
