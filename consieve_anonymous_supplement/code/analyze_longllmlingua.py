#!/usr/bin/env python3
"""Aggregate M4.5 LongLLMLingua results and frozen M4 comparators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import compute_f1, dedupe_answers, extract_answers
from scripts.aggregate_results import aggregate


def load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected list: {path}")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def all_row(path: Path) -> dict:
    records = load(path)
    row = aggregate(records)
    if any(row[key] != 406 for key in ("n_total", "n_qa", "n_token", "n_judge")):
        raise ValueError(f"incomplete QA artifact {path}: {row}")
    if row["n_provider_errors"] or row["n_judge_failures"]:
        raise ValueError(f"provider/judge failures in {path}")
    return row


def per_f1(record: dict) -> float:
    refs = dedupe_answers(extract_answers(record))
    return max((compute_f1(str(record.get("model_answer", "")), ref) for ref in refs), default=0.0)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def paired_bootstrap(ds_path: Path, baseline_path: Path, seed: int, iterations: int) -> dict:
    ds = {str(row["id"]): row for row in load(ds_path)}
    baseline = {str(row["id"]): row for row in load(baseline_path)}
    if set(ds) != set(baseline) or len(ds) != 406:
        raise ValueError("paired coverage mismatch")
    ids = list(ds)
    differences = {
        "f1": [per_f1(ds[i]) - per_f1(baseline[i]) for i in ids],
        "judge": [float(bool(ds[i].get("judge_correct"))) - float(bool(baseline[i].get("judge_correct"))) for i in ids],
    }
    rng = random.Random(seed)
    out = {}
    for metric, diffs in differences.items():
        draws = [sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(iterations)]
        less = sum(v <= 0 for v in draws) / iterations
        greater = sum(v >= 0 for v in draws) / iterations
        out[metric] = {
            "difference": statistics.mean(diffs),
            "ci95": [percentile(draws, .025), percentile(draws, .975)],
            "p_two_sided": min(1.0, 2 * min(less, greater)),
        }
    return {
        "schema_version": "m45_paired_bootstrap_v1",
        "comparison": "DS-ModernBERT minus LongLLMLingua",
        "scenario": "BM25@25",
        "budget": 256,
        "seed": seed,
        "iterations": iterations,
        "n_pairs": 406,
        "evidence_recall": "N/A — no exact sentence mapping for LongLLMLingua",
        "ds_raw": str(ds_path),
        "baseline_raw": str(baseline_path),
        "metrics": out,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--natural", type=Path, required=True)
    p.add_argument("--budget256", type=Path, required=True)
    p.add_argument("--m4-natural", type=Path, required=True)
    p.add_argument("--m4-matched", type=Path, required=True)
    p.add_argument("--raw-bm25", type=Path, required=True)
    p.add_argument("--ds-budget256", type=Path, required=True)
    p.add_argument("--efficiency-manifest", type=Path, required=True)
    p.add_argument("--m4-efficiency", type=Path, required=True)
    p.add_argument("--api-summary", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--iterations", type=int, default=10000)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    m4_natural = [row for row in csv.DictReader(args.m4_natural.open(encoding="utf-8")) if row["scenario"] == "BM25@25"]
    wanted_natural = {"OpenProvence", "DS-ModernBERT", "EXIT"}
    natural_rows = [row for row in m4_natural if row["method"] in wanted_natural]
    raw = all_row(args.raw_bm25)
    natural_rows.insert(0, {
        "method": "Raw BM25@25", "scenario": "BM25@25", "operating_point": "uncompressed",
        **raw, "raw_output": str(args.raw_bm25),
    })
    long_natural = all_row(args.natural)
    long_natural.update({
        "method": "LongLLMLingua", "scenario": "BM25@25",
        "operating_point": "official RAG example target_token=500",
        "evidence_precision": "N/A", "evidence_recall": "N/A",
        "evidence_note": "token-level output cannot be exactly mapped to frozen sentence labels",
        "raw_output": str(args.natural),
    })
    natural_rows.append(long_natural)
    write_csv(args.output_dir / "m45_natural_results.csv", natural_rows)

    m4_matched = [row for row in csv.DictReader(args.m4_matched.open(encoding="utf-8")) if row["scenario"] == "BM25@25" and row["budget"] == "256"]
    wanted_matched = {"OpenProvence", "DS-ModernBERT", "EXIT"}
    matched_rows = [row for row in m4_matched if row["method"] in wanted_matched]
    long_budget = all_row(args.budget256)
    records_budget = load(args.budget256)
    violations = [row["id"] for row in records_budget if int(row["context_tokens"]) > 256]
    if violations:
        raise ValueError(f"LongLLMLingua budget violations: {violations[:5]}")
    long_budget.update({
        "method": "LongLLMLingua", "method_key": "longllmlingua", "scenario": "BM25@25", "budget": 256,
        "fairness_mode": "per_query_hard_cap", "evidence_precision": "N/A", "evidence_recall": "N/A",
        "evidence_note": "token-level output cannot be exactly mapped to frozen sentence labels",
        "mean_compression_passes": statistics.mean(float(row["compression_passes"]) for row in records_budget),
        "max_compression_passes": max(int(row["compression_passes"]) for row in records_budget),
        "raw_output": str(args.budget256),
    })
    matched_rows.append(long_budget)
    write_csv(args.output_dir / "m45_budget256_results.csv", matched_rows)

    bootstrap = paired_bootstrap(args.ds_budget256, args.budget256, args.seed, args.iterations)
    (args.output_dir / "m45_bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")

    efficiency_manifest = json.loads(args.efficiency_manifest.read_text(encoding="utf-8"))
    m4_eff = list(csv.DictReader(args.m4_efficiency.open(encoding="utf-8")))
    efficiency_rows = [row for row in m4_eff if row["method"] in {"OpenProvence", "DS-ModernBERT", "EXIT"}]
    for mode, label in [("natural", "LongLLMLingua natural"), ("budget256", "LongLLMLingua @256 effective")]:
        summary = efficiency_manifest["modes"][mode]
        efficiency_rows.append({
            "method": label,
            "status": "reproduced",
            "parameter_count": efficiency_manifest["parameter_count"],
            "mean_latency_seconds": summary["mean_latency_seconds"],
            "median_latency_seconds": summary["median_latency_seconds"],
            "p95_latency_seconds": summary["p95_latency_seconds"],
            "throughput_queries_per_second": summary["throughput_queries_per_second"],
            "peak_vram_bytes": efficiency_manifest["peak_gpu_memory_bytes"],
            "warmup_queries": efficiency_manifest["warmup_queries"],
            "measured_queries": summary["measured_queries"],
            "batch_size": 1,
            "cuda_visible_devices": efficiency_manifest["cuda_visible_devices"],
            "torch_version": efficiency_manifest["torch_version"],
            "python_version": efficiency_manifest["python_version"],
            "transformers_version": efficiency_manifest["transformers_version"],
            "gpu_name": efficiency_manifest["gpu_name"],
            "gpu_total_memory_bytes": efficiency_manifest["gpu_total_memory_bytes"],
            "model_revision": efficiency_manifest["model_revision"],
            "mean_compression_passes": summary["mean_passes"],
            "raw_manifest": str(args.efficiency_manifest),
        })
    write_csv(args.output_dir / "m45_efficiency.csv", efficiency_rows)

    all_frontier_rows = list(csv.DictReader(args.m4_matched.open(encoding="utf-8")))
    all_frontier_rows.append({
        "method": "LongLLMLingua", "budget": "256", "context_tokens": str(long_budget["context_tokens"]),
        "f1": str(long_budget["f1"]),
    })
    points = [row for row in all_frontier_rows if row.get("scenario", "BM25@25") == "BM25@25"]
    frontier = []
    for row in points:
        tokens, f1 = float(row["context_tokens"]), float(row["f1"])
        dominated = any(
            float(other["context_tokens"]) <= tokens and float(other["f1"]) >= f1
            and (float(other["context_tokens"]) < tokens or float(other["f1"]) > f1)
            for other in points if other is not row
        )
        if not dominated:
            frontier.append({"method": row["method"], "budget": int(row["budget"]), "mean_tokens": tokens, "f1": f1})
    (args.output_dir / "m45_f1_token_pareto.json").write_text(json.dumps({
        "scenario": "BM25@25", "points": len(points), "frontier": frontier,
        "ds_on_frontier": any(str(row["method"]).startswith("DS-") for row in frontier),
    }, indent=2) + "\n", encoding="utf-8")

    api_summary = json.loads(args.api_summary.read_text(encoding="utf-8")) if args.api_summary else None
    completion = {
        "schema_version": "m45_completion_checks_v1",
        "status": "pass",
        "natural_samples": long_natural["n_total"],
        "budget256_samples": long_budget["n_total"],
        "budget256_cap_violations": len(violations),
        "generation_provider_errors": long_natural["n_provider_errors"] + long_budget["n_provider_errors"],
        "judge_failures": long_natural["n_judge_failures"] + long_budget["n_judge_failures"],
        "api_attempts": api_summary["calls"] if api_summary else None,
        "api_success_attempts": api_summary["status"].get("success", 0) if api_summary else None,
        "api_error_attempts": api_summary["status"].get("error", 0) if api_summary else None,
        "api_total_tokens_on_success": api_summary["by_model"]["qwen-plus"]["total_tokens"] if api_summary else None,
        "bootstrap_pairs": bootstrap["n_pairs"],
        "efficiency_measured_queries_each_mode": efficiency_manifest["modes"]["natural"]["measured_queries"],
        "stress": "not_applicable_without_new_alignment_heuristic",
        "bm25_top5": "skipped_as_optional_to_avoid_delaying_M5",
    }
    (args.output_dir / "m45_completion_checks.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"natural": long_natural, "budget256": long_budget, "bootstrap": bootstrap, "frontier": frontier}, indent=2))


if __name__ == "__main__":
    main()
