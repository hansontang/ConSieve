#!/usr/bin/env python3
"""Fail-closed Main1500 aggregation, paired statistics, and stability analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import compute_exact_match, compute_f1, dedupe_answers, extract_answers


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "evaluation/results/clean_v2_m5_main1500"
EXPECTED_COUNTS = {"All": 1500, "StreamingQA": 1294, "TimeQA": 206}
SEED = 20260810
ITERATIONS = 10000

M5_NATURAL = {
    "Raw BM25@25": RESULT_DIR / "m5_raw_bm25_top25_natural.json",
    "OpenProvence": RESULT_DIR / "m5_openprovence_natural.json",
    "DS-ModernBERT": RESULT_DIR / "m5_ds_modernbert_natural.json",
    "EXIT": RESULT_DIR / "m5_exit_natural.json",
    "LongLLMLingua": RESULT_DIR / "m5_longllmlingua_natural.json",
}
M5_BUDGET = {
    "OpenProvence": RESULT_DIR / "m5_openprovence_budget256.json",
    "DS-ModernBERT": RESULT_DIR / "m5_ds_modernbert_budget256.json",
    "EXIT": RESULT_DIR / "m5_exit_budget256.json",
    "LongLLMLingua": RESULT_DIR / "m5_longllmlingua_budget256.json",
}
GT_NATURAL = {
    "Raw BM25@25": ROOT / "evaluation/results/clean_v2_canonical_gt406/bm25_top25_20260811_070428.json",
    "OpenProvence": ROOT / "evaluation/results/clean_v2_canonical_gt406/bm25_top25_highlighter_open-provence-reranker-large-v1_20260811_070428.json",
    "DS-ModernBERT": ROOT / "evaluation/results/clean_v2_canonical_gt406/bm25_top25_highlighter_ts-highlighter-modernbert-timed-20260119_022815_20260811_070428.json",
    "EXIT": ROOT / "evaluation/results/clean_v2_m4_gt406/m4_exit_natural_top25.json",
    "LongLLMLingua": ROOT / "evaluation/results/clean_v2_m45_gt406/m45_longllmlingua_natural.json",
}
GT_BUDGET = {
    "OpenProvence": ROOT / "evaluation/results/clean_v2_m4_gt406/m4_openprovence_budget_256.json",
    "DS-ModernBERT": ROOT / "evaluation/results/clean_v2_m4_gt406/m4_ds_modernbert_budget_256.json",
    "EXIT": ROOT / "evaluation/results/clean_v2_m4_gt406/m4_exit_budget_256.json",
    "LongLLMLingua": ROOT / "evaluation/results/clean_v2_m45_gt406/m45_longllmlingua_budget_256.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected per-query list: {path}")
    return value


def subset(records: list[dict], source: str) -> list[dict]:
    if source == "All":
        return records
    prefix = "streamingqa" if source == "StreamingQA" else "timeqa"
    return [row for row in records if str(row.get("source", "")).startswith(prefix)]


def references(record: dict) -> list[str]:
    return dedupe_answers(extract_answers(record))


def per_f1(record: dict) -> float:
    refs = references(record)
    return max((compute_f1(str(record.get("model_answer", "")), ref) for ref in refs), default=0.0)


def per_em(record: dict) -> float:
    refs = references(record)
    return max((compute_exact_match(str(record.get("model_answer", "")), ref) for ref in refs), default=0.0)


def validate(records: list[dict], path: Path, budget: int | None = None) -> None:
    manifest = json.loads((ROOT / "manifests/main1500.json").read_text(encoding="utf-8"))
    expected_ids = [str(row["id"]) for row in manifest["samples"]]
    actual_ids = [str(row["id"]) for row in records]
    if actual_ids != expected_ids or len(set(actual_ids)) != 1500:
        raise ValueError(f"Main1500 denominator/order failure: {path}")
    counts = Counter(str(row.get("source", "")) for row in records)
    if dict(counts) != {"streamingqa": 1294, "timeqa": 206}:
        raise ValueError(f"source denominator failure: {path}: {counts}")
    if any(row.get("generation_status") != "success" for row in records):
        raise ValueError(f"unrecovered generation failure: {path}")
    if any(row.get("judge_status") != "judged" for row in records):
        raise ValueError(f"unrecovered Judge failure: {path}")
    if budget is not None:
        violations = [row["id"] for row in records if int(row["context_tokens"]) > budget]
        if violations:
            raise ValueError(f"{path}: {len(violations)} budget violations: {violations[:5]}")


def aggregate(records: list[dict], method: str, source: str, operating_point: str, path: Path) -> dict:
    n = len(records)
    tokens = [int(row["context_tokens"]) for row in records]
    ratios = [float(row.get("compression_ratio", 1.0)) for row in records]
    generation_failures = sum(row.get("generation_status") != "success" for row in records)
    judge_failures = sum(row.get("judge_status") != "judged" for row in records)
    content_filtered = sum(str(row.get("model_answer", "")).strip() == "[CONTENT_FILTERED]" for row in records)
    return {
        "method": method,
        "source": source,
        "n": n,
        "scenario": "BM25@25",
        "operating_point": operating_point,
        "f1": statistics.mean(per_f1(row) for row in records),
        "em": statistics.mean(per_em(row) for row in records),
        "judge": statistics.mean(float(bool(row.get("judge_correct"))) for row in records),
        "mean_context_tokens": statistics.mean(tokens),
        "median_context_tokens": statistics.median(tokens),
        "compression_ratio": statistics.mean(ratios),
        "empty_context_rate": sum(not str(row.get("context", "")).strip() for row in records) / n,
        "generation_failure_rate": generation_failures / n,
        "judge_failure_rate": judge_failures / n,
        "content_filter_rate": content_filtered / n,
        "evidence_precision": "N/A",
        "evidence_recall": "N/A",
        "evidence_note": "Main1500 has no frozen sentence-evidence sidecar; no fuzzy/new alignment introduced",
        "generation_input_tokens": sum(int(row.get("input_tokens") or 0) for row in records),
        "generation_output_tokens": sum(int(row.get("output_tokens") or 0) for row in records),
        "raw_output": str(path.relative_to(ROOT)),
        "raw_output_sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_rows(ds: list[dict], base: list[dict], comparison: str, baseline: str) -> list[dict]:
    output = []
    for source in EXPECTED_COUNTS:
        ds_part = subset(ds, source)
        base_by_id = {str(row["id"]): row for row in subset(base, source)}
        if len(ds_part) != EXPECTED_COUNTS[source] or set(base_by_id) != {str(row["id"]) for row in ds_part}:
            raise ValueError(f"paired coverage failure: {comparison} {source}")
        metrics = {
            "f1": np.asarray([per_f1(row) - per_f1(base_by_id[str(row["id"])]) for row in ds_part]),
            "judge": np.asarray([
                float(bool(row.get("judge_correct")))
                - float(bool(base_by_id[str(row["id"])].get("judge_correct")))
                for row in ds_part
            ]),
        }
        rng = np.random.default_rng(SEED)
        indices = rng.integers(0, len(ds_part), size=(ITERATIONS, len(ds_part)))
        for metric, differences in metrics.items():
            draws = differences[indices].mean(axis=1)
            output.append({
                "comparison": comparison,
                "baseline": baseline,
                "source": source,
                "metric": metric,
                "n_pairs": len(ds_part),
                "iterations": ITERATIONS,
                "seed": SEED,
                "mean_difference": float(differences.mean()),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "p_two_sided": float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
                "status": "computed",
            })
    if baseline in {"EXIT", "OpenProvence"}:
        output.append({
            "comparison": comparison,
            "baseline": baseline,
            "source": "All",
            "metric": "evidence_recall",
            "n_pairs": 1500,
            "iterations": ITERATIONS,
            "seed": SEED,
            "mean_difference": "N/A",
            "ci95_low": "N/A",
            "ci95_high": "N/A",
            "p_two_sided": "N/A",
            "status": "not_applicable_no_frozen_main1500_sentence_evidence",
        })
    return output


def effect(records: dict[str, list[dict]], method: str, baseline: str, metric: str, source: str) -> float:
    left = subset(records[method], source)
    right = subset(records[baseline], source)
    if metric == "f1":
        return statistics.mean(per_f1(row) for row in left) - statistics.mean(per_f1(row) for row in right)
    if metric == "judge":
        return statistics.mean(float(bool(row["judge_correct"])) for row in left) - statistics.mean(float(bool(row["judge_correct"])) for row in right)
    if metric == "context_reduction":
        return 1.0 - statistics.mean(int(row["context_tokens"]) for row in left) / statistics.mean(int(row["context_tokens"]) for row in right)
    raise ValueError(metric)


def sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def stability_rows(gt_natural: dict[str, list[dict]], gt_budget: dict[str, list[dict]], m5_natural: dict[str, list[dict]], m5_budget: dict[str, list[dict]]) -> list[dict]:
    specs = [
        ("Natural", "DS-ModernBERT", "OpenProvence", "f1", gt_natural, m5_natural),
        ("Natural", "DS-ModernBERT", "EXIT", "f1", gt_natural, m5_natural),
        ("Natural", "DS-ModernBERT", "LongLLMLingua", "f1", gt_natural, m5_natural),
        ("Natural", "DS-ModernBERT", "Raw BM25@25", "f1", gt_natural, m5_natural),
        ("Natural", "DS-ModernBERT", "Raw BM25@25", "context_reduction", gt_natural, m5_natural),
        ("@256", "DS-ModernBERT", "OpenProvence", "f1", gt_budget, m5_budget),
        ("@256", "DS-ModernBERT", "EXIT", "f1", gt_budget, m5_budget),
        ("@256", "DS-ModernBERT", "LongLLMLingua", "f1", gt_budget, m5_budget),
        ("@256", "DS-ModernBERT", "EXIT", "judge", gt_budget, m5_budget),
        ("@256", "DS-ModernBERT", "LongLLMLingua", "judge", gt_budget, m5_budget),
    ]
    rows = []
    for point, method, baseline, metric, gt, main in specs:
        main_source_effects = {
            source: effect(main, method, baseline, metric, source)
            for source in ("StreamingQA", "TimeQA")
        }
        source_reversal = sign(main_source_effects["StreamingQA"]) * sign(main_source_effects["TimeQA"]) < 0
        for source in EXPECTED_COUNTS:
            gt_value = effect(gt, method, baseline, metric, source)
            main_value = effect(main, method, baseline, metric, source)
            ratio = abs(main_value) / abs(gt_value) if abs(gt_value) > 1e-12 else None
            rows.append({
                "operating_point": point,
                "comparison": f"{method} minus {baseline}",
                "metric": metric,
                "source": source,
                "gt406_effect": gt_value,
                "main1500_effect": main_value,
                "direction_preserved": sign(gt_value) == sign(main_value),
                "effect_magnitude_ratio_abs": ratio,
                "same_order_of_magnitude": ratio is not None and 0.25 <= ratio <= 4.0,
                "main1500_source_specific_reversal": source_reversal,
            })
    return rows


def main() -> None:
    natural_records = {method: load(path) for method, path in M5_NATURAL.items()}
    budget_records = {method: load(path) for method, path in M5_BUDGET.items()}
    for method, path in M5_NATURAL.items():
        validate(natural_records[method], path)
    for method, path in M5_BUDGET.items():
        validate(budget_records[method], path, budget=256)

    natural_rows = []
    natural_op = {
        "Raw BM25@25": "uncompressed ordered canonical top25",
        "OpenProvence": "M1 validation-frozen threshold=0.20",
        "DS-ModernBERT": "M1 validation-frozen threshold=0.40",
        "EXIT": "official threshold=0.5 inclusive; M4 frozen overflow handling",
        "LongLLMLingua": "official RAG target_token=500; M4.5 frozen long-input adaptation",
    }
    for method in M5_NATURAL:
        for source, expected in EXPECTED_COUNTS.items():
            part = subset(natural_records[method], source)
            if len(part) != expected:
                raise ValueError(f"natural subgroup denominator failure: {method} {source}")
            natural_rows.append(aggregate(part, method, source, natural_op[method], M5_NATURAL[method]))
    write_csv(ROOT / "reports/m5_main1500_natural.csv", natural_rows)

    budget_rows = []
    for method in M5_BUDGET:
        for source, expected in EXPECTED_COUNTS.items():
            part = subset(budget_records[method], source)
            if len(part) != expected:
                raise ValueError(f"budget subgroup denominator failure: {method} {source}")
            row = aggregate(part, method, source, "per-query hard cap=256 Qwen context tokens", M5_BUDGET[method])
            row["budget"] = 256
            row["hard_cap_violations"] = sum(int(item["context_tokens"]) > 256 for item in part)
            budget_rows.append(row)
    write_csv(ROOT / "reports/m5_main1500_budget256.csv", budget_rows)

    bootstrap = []
    for baseline in ("EXIT", "LongLLMLingua", "OpenProvence"):
        bootstrap.extend(
            bootstrap_rows(
                budget_records["DS-ModernBERT"],
                budget_records[baseline],
                f"DS-ModernBERT minus {baseline}",
                baseline,
            )
        )
    write_csv(ROOT / "reports/m5_main1500_bootstrap.csv", bootstrap)

    gt_natural = {method: load(path) for method, path in GT_NATURAL.items()}
    gt_budget = {method: load(path) for method, path in GT_BUDGET.items()}
    stability = stability_rows(gt_natural, gt_budget, natural_records, budget_records)
    write_csv(ROOT / "reports/m5_gt406_main1500_stability.csv", stability)

    practical = [row for row in natural_rows if row["source"] == "All"] + [
        row for row in budget_rows if row["source"] == "All"
    ]
    write_csv(ROOT / "reports/m5_final_practical_results.csv", practical)

    artifact_paths = [
        ROOT / "reports/m5_main1500_natural.csv",
        ROOT / "reports/m5_main1500_budget256.csv",
        ROOT / "reports/m5_main1500_bootstrap.csv",
        ROOT / "reports/m5_gt406_main1500_stability.csv",
        ROOT / "reports/m5_final_practical_results.csv",
    ]
    completion = {
        "schema_version": "m5_analysis_completion_v1",
        "status": "pass",
        "natural_rows": len(natural_rows),
        "budget256_rows": len(budget_rows),
        "bootstrap_rows": len(bootstrap),
        "stability_rows": len(stability),
        "practical_rows": len(practical),
        "main1500_denominator": 1500,
        "source_denominators": {"streamingqa": 1294, "timeqa": 206},
        "evidence_metrics": "N/A — no frozen Main1500 sentence-evidence sidecar",
        "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in artifact_paths},
        "raw_qa_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in list(M5_NATURAL.values()) + list(M5_BUDGET.values())
        },
    }
    (ROOT / "reports/m5_analysis_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
