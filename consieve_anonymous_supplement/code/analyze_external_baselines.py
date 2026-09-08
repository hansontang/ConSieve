#!/usr/bin/env python3
"""Aggregate M4 natural/matched results, draw curves, and bootstrap DS vs EXIT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import compute_f1, dedupe_answers, extract_answers
from scripts.aggregate_results import aggregate


ALIASES = {
    "open-provence-reranker-large-v1": "OpenProvence",
    "semantic-highlight-bilingual-v1": "General Zilliz",
    "ts-highlighter-modernbert-timed-20260119_022815": "DS-ModernBERT",
    "ts-highlighter-zilliz-timed-20260119_011422": "DS-Zilliz",
    "openprovence": "OpenProvence",
    "general_zilliz": "General Zilliz",
    "ds_modernbert": "DS-ModernBERT",
    "ds_zilliz": "DS-Zilliz",
    "exit": "EXIT",
}


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def alias_from_name(name: str) -> str:
    for key, value in ALIASES.items():
        if key in name:
            return value
    return name


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def per_record_values(records: list[dict], metric: str) -> np.ndarray:
    values = []
    for record in records:
        if metric == "f1":
            refs = dedupe_answers(extract_answers(record))
            values.append(max(compute_f1(record.get("model_answer", ""), ref) for ref in refs))
        elif metric == "judge":
            values.append(float(bool(record.get("judge_correct"))))
        elif metric == "evidence_recall":
            tp, fn = int(record.get("evidence_tp", 0)), int(record.get("evidence_fn", 0))
            values.append(tp / (tp + fn) if tp + fn else 0.0)
        else:
            raise ValueError(metric)
    return np.asarray(values, dtype=float)


def validate_complete(row: dict, path: Path) -> None:
    expected = 406
    for key in ("n_total", "n_qa", "n_token", "n_judge", "n_evidence"):
        if int(row.get(key, -1)) != expected:
            raise ValueError(f"{path}: {key}={row.get(key)}, expected {expected}")
    if int(row.get("n_provider_errors", -1)) or int(row.get("n_judge_failures", -1)):
        raise ValueError(
            f"{path}: provider_errors={row.get('n_provider_errors')}, "
            f"judge_failures={row.get('n_judge_failures')}"
        )


def bootstrap(ds: list[dict], baseline: list[dict], iterations: int, seed: int) -> dict:
    by_id = {str(row["id"]): row for row in baseline}
    ds_ordered = [row for row in ds if str(row["id"]) in by_id]
    other = [by_id[str(row["id"])] for row in ds_ordered]
    if len(ds_ordered) != 406:
        raise ValueError(f"paired coverage is {len(ds_ordered)}, expected 406")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ds_ordered), size=(iterations, len(ds_ordered)))
    output = {"iterations": iterations, "seed": seed, "n_pairs": len(ds_ordered), "metrics": {}}
    for metric in ("f1", "judge", "evidence_recall"):
        differences = per_record_values(ds_ordered, metric) - per_record_values(other, metric)
        samples = differences[indices].mean(axis=1)
        output["metrics"][metric] = {
            "difference": float(differences.mean()),
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
            "p_two_sided": float(min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0)))),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-dir", type=Path, required=True)
    parser.add_argument("--m4-qa-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--representative-budget", type=int, default=512)
    parser.add_argument("--efficiency", type=Path)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    natural_rows = []
    raw_context_tokens_by_top_k = {}
    latest_stamp = "20260811_070428"
    for top_k in (25, 5):
        raw_context_path = args.m1_dir / f"bm25_top{top_k}_{latest_stamp}.json"
        raw_context_tokens = {str(item["id"]): int(item["context_tokens"]) for item in load(raw_context_path)}
        raw_context_tokens_by_top_k[top_k] = raw_context_tokens
        for filename in args.m1_dir.glob(f"bm25_top{top_k}_highlighter_*_{latest_stamp}.json"):
            records = load(filename)
            row = aggregate(records)
            validate_complete(row, filename)
            row["compression_ratio"] = sum(
                int(item["context_tokens"]) / raw_context_tokens[str(item["id"])]
                for item in records if raw_context_tokens[str(item["id"])]
            ) / len(records)
            row.update({"method": alias_from_name(filename.stem), "scenario": f"BM25@{top_k}", "operating_point": "M1 validation-frozen threshold", "raw_output": str(filename)})
            natural_rows.append(row)
    for path in sorted(args.m4_qa_dir.glob("m4_exit_natural*.json")):
        records = load(path)
        row = aggregate(records)
        validate_complete(row, path)
        top_k = 5 if "top5" in path.stem else 25
        scenario = f"BM25@{top_k}"
        raw_context_tokens = raw_context_tokens_by_top_k[top_k]
        row["compression_ratio"] = sum(
            int(item["context_tokens"]) / raw_context_tokens[str(item["id"])]
            for item in records if raw_context_tokens[str(item["id"])]
        ) / len(records)
        row.update({"method": "EXIT", "scenario": scenario, "operating_point": "official threshold 0.5", "raw_output": str(path)})
        natural_rows.append(row)
    efficiency = {}
    if args.efficiency and args.efficiency.exists():
        with args.efficiency.open(encoding="utf-8") as handle:
            efficiency = {row["method"]: row for row in csv.DictReader(handle)}
    for row in natural_rows:
        item = efficiency.get(row["method"], {})
        row["compressor_parameters"] = item.get("parameter_count")
        row["peak_vram_bytes"] = item.get("peak_vram_bytes")
        row["benchmark_median_latency_seconds"] = item.get("median_latency_seconds")
        row["compression_ratio_definition"] = "mean per-query retained context tokens / raw BM25 context tokens"
    if len(natural_rows) != 10:
        raise ValueError(f"natural table has {len(natural_rows)} rows, expected 10 (five methods x two top-k)")
    write_csv(args.reports_dir / "m4_natural_results.csv", natural_rows)

    matched_rows = []
    raw_by_key = {}
    for path in sorted(args.m4_qa_dir.glob("m4_*_budget_*.json")):
        records = load(path)
        stem = path.stem
        budget = int(stem.rsplit("_", 1)[-1])
        method_key = stem[len("m4_") : stem.rfind("_budget_")]
        row = aggregate(records)
        validate_complete(row, path)
        cap_violations = [str(record["id"]) for record in records if int(record["context_tokens"]) > budget]
        if cap_violations:
            raise ValueError(f"{path}: {len(cap_violations)} hard-cap violations: {cap_violations[:5]}")
        raw_context_tokens = raw_context_tokens_by_top_k[25]
        row["compression_ratio"] = sum(
            int(item["context_tokens"]) / raw_context_tokens[str(item["id"])]
            for item in records if raw_context_tokens[str(item["id"])]
        ) / len(records)
        row.update({"method": alias_from_name(method_key), "method_key": method_key, "budget": budget, "scenario": "BM25@25", "fairness_mode": "per_query_hard_cap", "raw_output": str(path)})
        item = efficiency.get(row["method"], {})
        row["compressor_parameters"] = item.get("parameter_count")
        row["peak_vram_bytes"] = item.get("peak_vram_bytes")
        row["benchmark_median_latency_seconds"] = item.get("median_latency_seconds")
        row["compression_ratio_definition"] = "mean per-query retained context tokens / raw BM25 context tokens"
        matched_rows.append(row)
        raw_by_key[(method_key, budget)] = records
    if raw_by_key:
        reference_ids = None
        for key, records in raw_by_key.items():
            ids = [str(record["id"]) for record in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate IDs in matched output {key}")
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise ValueError(f"paired ID/order mismatch in matched output {key}")
        budgets = sorted({budget for _, budget in raw_by_key})
        methods = {method for method, _ in raw_by_key}
        expected_methods = {"openprovence", "general_zilliz", "ds_modernbert", "ds_zilliz", "exit"}
        if methods != expected_methods or len(raw_by_key) != len(budgets) * len(expected_methods):
            raise ValueError(
                f"matched matrix incomplete: methods={sorted(methods)}, budgets={budgets}, "
                f"cells={len(raw_by_key)}"
            )
    write_csv(args.reports_dir / "m4_matched_budget.csv", matched_rows)
    if matched_rows:
        frontier = []
        for candidate in matched_rows:
            dominated = any(
                other is not candidate
                and float(other["context_tokens"]) <= float(candidate["context_tokens"])
                and float(other["f1"]) >= float(candidate["f1"])
                and (
                    float(other["context_tokens"]) < float(candidate["context_tokens"])
                    or float(other["f1"]) > float(candidate["f1"])
                )
                for other in matched_rows
            )
            if not dominated:
                frontier.append({
                    "method": candidate["method"], "method_key": candidate["method_key"],
                    "budget": candidate["budget"], "actual_mean_tokens": candidate["context_tokens"],
                    "f1": candidate["f1"],
                })
        (args.reports_dir / "m4_f1_token_pareto.json").write_text(
            json.dumps({"scenario": "BM25@25", "quality": "F1", "cost": "actual mean context tokens", "frontier": frontier}, indent=2) + "\n",
            encoding="utf-8",
        )

    ds = raw_by_key.get(("ds_modernbert", args.representative_budget))
    exit_rows = raw_by_key.get(("exit", args.representative_budget))
    if ds is not None and exit_rows is not None:
        result = bootstrap(ds, exit_rows, args.iterations, args.seed)
        result.update({"comparison": "DS-ModernBERT minus EXIT", "scenario": "BM25@25", "budget": args.representative_budget})
        (args.reports_dir / "m4_bootstrap_ds_modernbert_vs_exit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if matched_rows:
        import matplotlib.pyplot as plt

        args.figures_dir.mkdir(parents=True, exist_ok=True)
        for metric, label, filename in [
            ("f1", "F1", "m4_f1_vs_context_tokens.png"),
            ("judge", "Judge Accuracy", "m4_judge_vs_context_tokens.png"),
            ("evidence_recall", "Evidence Recall", "m4_evidence_recall_vs_context_tokens.png"),
        ]:
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            annotation_offsets = {
                "DS-ModernBERT": (4, -12),
                "DS-Zilliz": (4, 5),
                "EXIT": (4, 5),
                "General Zilliz": (4, -12),
                "OpenProvence": (4, 5),
            }
            for method in sorted({row["method"] for row in matched_rows}):
                points = sorted((row for row in matched_rows if row["method"] == method), key=lambda row: row["context_tokens"])
                ax.plot([row["context_tokens"] for row in points], [row[metric] for row in points], marker="o", label=method)
                for row in points:
                    offset = annotation_offsets[method]
                    if method == "DS-ModernBERT" and int(row["budget"]) == 128:
                        offset = (4, 5)
                    if method == "DS-Zilliz" and int(row["budget"]) == 128:
                        offset = (4, -12)
                    ax.annotate(
                        str(row["budget"]),
                        (row["context_tokens"], row[metric]),
                        fontsize=7,
                        xytext=offset,
                        textcoords="offset points",
                    )
            ax.set_xscale("log")
            ax.set_xlabel("Actual mean context tokens/query (log scale)")
            ax.set_ylabel(label)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(args.figures_dir / filename, dpi=200)
            plt.close(fig)
    print(json.dumps({"natural_rows": len(natural_rows), "matched_rows": len(matched_rows)}, indent=2))


if __name__ == "__main__":
    main()
