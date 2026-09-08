#!/usr/bin/env python3
"""Aggregate frozen M2 scores, paired bootstrap CIs, cases, and taxonomy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable


SEED = 20260810
MODELS = ["openprovence", "ds_modernbert", "general_zilliz", "ds_zilliz"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels); negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for pos in range(cursor, end):
            ranks[ordered[pos]] = rank
        cursor = end
    rank_sum = sum(ranks[i] for i, label in enumerate(labels) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    ordered = sorted(range(len(scores)), key=lambda i: (-scores[i], stable(str(i))))
    tp = 0; total = 0.0
    for rank, index in enumerate(ordered, 1):
        if labels[index]:
            tp += 1; total += tp / rank
    return total / positives


def pair_metrics(row: dict[str, Any]) -> dict[str, float]:
    indices = [index - 1 for index in row["positive_evidence_indices"] if 1 <= index <= len(row["positive_scores"])]
    positive = [float(row["positive_scores"][i]) for i in indices]
    negative = [float(row["perturbed_scores"][i]) for i in indices]
    threshold = float(row["threshold"])
    positive_recall = safe_mean([float(score > threshold) for score in positive])
    invalid_retention = safe_mean([float(score > threshold) for score in negative])
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
        "evidence_count": len(indices),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    for key in [
        "positive_evidence_recall", "invalid_evidence_retention", "constraint_rejection_rate", "score_gap_mean",
        "score_gap_median", "positive_gap_rate", "pairwise_discrimination_accuracy", "positive_empty_rate",
        "perturbed_empty_rate", "constraint_sensitivity_delta",
    ]:
        result[key] = safe_mean([float(item[key]) for item in paired if item[key] is not None])
    result["auroc"] = auroc(labels, scores)
    result["auprc"] = average_precision(labels, scores)
    return result


def groups(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    output = [("overall", "all", rows)]
    for field in ("dimension", "source"):
        for value in sorted({str(row[field]) for row in rows}):
            output.append((field, value, [row for row in rows if str(row[field]) == value]))
    temporal = [row for row in rows if row["dimension"] == "temporal"]
    output.append(("primary", "temporal_all", temporal))
    for source in sorted({str(row["source"]) for row in temporal}):
        output.append(("primary", f"temporal_{source}", [row for row in temporal if str(row["source"]) == source]))
    return output


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position)); upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], metric: str,
    pair_ids: list[str], iterations: int = 10000,
) -> dict[str, Any]:
    values = []
    for pair_id in pair_ids:
        a = pair_metrics(left[pair_id]).get(metric); b = pair_metrics(right[pair_id]).get(metric)
        if a is not None and b is not None:
            values.append(float(a) - float(b))
    if not values:
        return {"n_pairs": 0, "estimate": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(SEED + int(stable(metric)[:8], 16))
    draws = []
    count = len(values)
    for _ in range(iterations):
        draws.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    return {"n_pairs": count, "estimate": mean(values), "ci95_low": percentile(draws, 0.025), "ci95_high": percentile(draws, 0.975)}


def evidence_excerpt(row: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    result = []
    for index in row["positive_evidence_indices"][:limit]:
        i = index - 1
        result.append(
            {
                "index": index, "sentence": row["sentences"][i],
                "q_positive_score": row["positive_scores"][i], "q_perturbed_score": row["perturbed_scores"][i],
                "q_positive_retained": row["positive_retained"][i], "q_perturbed_retained": row["perturbed_retained"][i],
            }
        )
    return result


def case_studies(
    score_maps: dict[str, dict[str, dict[str, Any]]], validity_path: Path, output_md: Path, output_html: Path,
) -> dict[str, int]:
    op = score_maps["openprovence"]; ds = score_maps["ds_modernbert"]
    shared = sorted(set(op) & set(ds))
    success, failure = [], []
    for pair_id in shared:
        o = pair_metrics(op[pair_id]); d = pair_metrics(ds[pair_id])
        if o["positive_evidence_recall"] and o["invalid_evidence_retention"] and d["positive_evidence_recall"] and d["invalid_evidence_retention"] < o["invalid_evidence_retention"]:
            success.append(pair_id)
        if d["positive_evidence_recall"] < 1 or d["invalid_evidence_retention"] > 0:
            failure.append(pair_id)
    success.sort(key=lambda x: stable(f"{SEED}:success:{x}"))
    failure.sort(key=lambda x: stable(f"{SEED}:failure:{x}"))
    validity = list(csv.DictReader(validity_path.open(encoding="utf-8-sig")))
    collisions = [row for row in validity if row["validity_label"].startswith("COLLISION")]
    collisions.sort(key=lambda row: stable(f"{SEED}:collision:{row['pair_id']}"))
    cases = []
    for category, selected in (("Successful minimal pairs", success[:5]), ("DS-ModernBERT failure cases", failure[:5])):
        for pair_id in selected:
            o = op[pair_id]; d = ds[pair_id]
            cases.append(
                {
                    "category": category, "pair_id": pair_id, "source": o["source"], "dimension": o["dimension"],
                    "original_query": next(row for row in validity if row["pair_id"] == pair_id)["original_query"],
                    "perturbed_query": next(row for row in validity if row["pair_id"] == pair_id)["perturbed_query"],
                    "openprovence": evidence_excerpt(o), "ds_modernbert": evidence_excerpt(d),
                }
            )
    for row in collisions[:5]:
        cases.append(
            {
                "category": "Perturbation collision cases", "pair_id": row["pair_id"], "source": row["source"], "dimension": row["dimension"],
                "original_query": row["original_query"], "perturbed_query": row["perturbed_query"],
                "positive_rationale": row["positive_rationale"], "perturbed_rationale": row["perturbed_rationale"],
                "perturbed_evidence_indices": row["perturbed_evidence_indices"],
            }
        )
    lines = ["# M2 deterministic case studies", "", f"Selection seed: `{SEED}`. Cases are selected by a hash ordering within predeclared behavior categories; no performance-based manual cherry-picking.", ""]
    for category in ("Successful minimal pairs", "DS-ModernBERT failure cases", "Perturbation collision cases"):
        lines.extend([f"## {category}", ""])
        for case in [item for item in cases if item["category"] == category]:
            lines.extend([f"### {case['pair_id']}", "", f"- Source/dimension: `{case['source']}` / `{case['dimension']}`", f"- q+: {case['original_query']}", f"- q−: {case['perturbed_query']}"])
            if "openprovence" in case:
                lines.append(f"- OpenProvence evidence scores: `{json.dumps(case['openprovence'], ensure_ascii=False)}`")
                lines.append(f"- DS-ModernBERT evidence scores: `{json.dumps(case['ds_modernbert'], ensure_ascii=False)}`")
            else:
                lines.append(f"- q+ annotation: {case['positive_rationale']}")
                lines.append(f"- q− collision annotation: {case['perturbed_rationale']}")
            lines.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    body = "\n".join(f"<pre>{html.escape(line)}</pre>" if line.startswith("- ") else f"<h2>{html.escape(line[3:])}</h2>" if line.startswith("## ") else f"<h3>{html.escape(line[4:])}</h3>" if line.startswith("### ") else f"<p>{html.escape(line)}</p>" for line in lines[1:])
    output_html.write_text(f"<!doctype html><meta charset='utf-8'><title>M2 case studies</title><style>body{{font:15px system-ui;max-width:1200px;margin:40px auto}}pre{{white-space:pre-wrap;background:#f6f8fa;padding:8px}}</style><h1>M2 deterministic case studies</h1>{body}", encoding="utf-8")
    return Counter(case["category"] for case in cases)


def core_question(value: str) -> str:
    value = " ".join(value.split()).strip().casefold()
    return re.sub(r"^today is [^.]+\.\s*", "", value)


def validation_taxonomy(supplemental: Path, validity_path: Path, output: Path) -> dict[str, Any]:
    if not supplemental.exists():
        return {"available": False, "reason": f"missing {supplemental}"}
    rows = [row for row in read_jsonl(supplemental) if row.get("annotation_expectation_match") is False]
    validity = list(csv.DictReader(validity_path.open(encoding="utf-8-sig")))
    by_question = {row["perturbed_query"].strip().lower(): row for row in validity}
    hard_negative = [row for row in rows if row.get("is_hard_negative")]
    positive = [row for row in rows if not row.get("is_hard_negative")]
    for pool in (hard_negative, positive):
        pool.sort(key=lambda row: stable(f"{SEED}:taxonomy:{row.get('id', row.get('qa_id', row.get('question', '')))}"))
    # Preserve both disagreement directions instead of allowing the much larger
    # hard-negative group to crowd out unsupported positive examples.
    selected = hard_negative[:80] + positive[:20]
    output_rows = []
    for row in selected:
        question = str(row.get("question", row.get("query", "")))
        matched = by_question.get(question.strip().lower())
        text = " ".join(row.get("original_sentences") or row.get("sentences") or row.get("texts") or [])
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
        is_negative = bool(row.get("is_hard_negative"))
        has_citations = bool(row.get("relevant_indices"))
        unchanged = core_question(question) == core_question(str(row.get("original_question", "")))
        if not is_negative and not has_citations:
            category = "positive-side missing evidence / annotator false negative"
        elif unchanged and is_negative:
            category = "query perturbation failure / no-op"
        elif matched and matched["validity_label"].startswith("COLLISION"):
            category = "genuine perturbation collision"
        elif matched and matched["validity_label"].startswith("INVALID"):
            category = "query perturbation failure"
        elif is_negative and has_citations and len(years) >= 2:
            category = "document contains multiple time periods/entities"
        elif row.get("annotation_parse_success") is False:
            category = "historical annotation bug"
        elif is_negative and has_citations:
            category = "genuine collision candidate (requires human review)"
        else:
            category = "other / metadata mismatch (requires human review)"
        output_rows.append(
            {
                "sample_id": row.get("id", row.get("qa_id", "")), "source": row.get("source", "unknown"), "question": question,
                "automatic_taxonomy": category, "m2_pair_match": bool(matched),
                "expected_hard_negative": is_negative, "annotation_answerable": row.get("answerable"), "annotation_indices": json.dumps(row.get("relevant_indices", [])),
                "human_taxonomy": "", "human_notes": "",
            }
        )
    if output_rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
    return {"available": True, "disagreements": len(rows), "audited_sample": len(output_rows), "automatic_categories": dict(Counter(row["automatic_taxonomy"] for row in output_rows)), "human_fields_pending": True}


def dimension_metadata_audit(candidates: Path, output_csv: Path, output_json: Path) -> dict[str, Any]:
    rows = read_jsonl(candidates)
    temporal_words = set("january february march april may june july august september october november december jan feb mar apr jun jul aug sep oct nov dec before after between during until since year years month months".split())
    output_rows = []
    for row in rows:
        added = [str(token).casefold().strip(".,?!:;()[]") for token in row["perturbed_constraint_delta"]["added"]]
        temporal_signal = bool(set(added) & temporal_words) or any(re.fullmatch(r"(?:18|19|20)\d{2}", token) for token in added)
        output_rows.append(
            {
                "pair_id": row["pair_id"], "recorded_dimension": row["dimension"],
                "added_tokens": " ".join(row["perturbed_constraint_delta"]["added"]),
                "removed_tokens": " ".join(row["perturbed_constraint_delta"]["removed"]),
                "explicit_temporal_delta_signal": temporal_signal,
                "recorded_non_temporal_but_temporal_delta": row["dimension"] != "temporal" and temporal_signal,
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
    summary = {"candidate_pairs": len(rows), "heuristic_definition": "added month/before-after/range token or 4-digit year", "by_recorded_dimension": {}}
    for dimension in sorted({row["dimension"] for row in rows}):
        subset = [row for row in output_rows if row["recorded_dimension"] == dimension]
        count = sum(row["explicit_temporal_delta_signal"] for row in subset)
        summary["by_recorded_dimension"][dimension] = {"pairs": len(subset), "temporal_delta_signal": count, "rate": count / len(subset)}
    summary["recorded_non_temporal_pairs"] = sum(row["recorded_dimension"] != "temporal" for row in output_rows)
    summary["recorded_non_temporal_with_temporal_delta"] = sum(row["recorded_non_temporal_but_temporal_delta"] for row in output_rows)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-dir", type=Path, default=root / "reports/m2_raw_scores")
    ap.add_argument("--validity", type=Path, default=root / "reports/m2_perturbation_validity.csv")
    ap.add_argument("--collision-summary", type=Path, default=root / "reports/m2_collision_summary.json")
    ap.add_argument("--results", type=Path, default=root / "reports/m2_stress_results.csv")
    ap.add_argument("--bootstrap", type=Path, default=root / "reports/m2_paired_bootstrap.csv")
    ap.add_argument("--case-md", type=Path, default=root / "reports/m2_case_studies.md")
    ap.add_argument("--case-html", type=Path, default=root / "reports/m2_case_studies.html")
    ap.add_argument("--taxonomy", type=Path, default=root / "reports/m2_validation_disagreement_taxonomy.csv")
    ap.add_argument("--supplemental", type=Path, default=root / "reports/thresholds/validation_labels_supplemental_qwen_plus_qc.jsonl")
    ap.add_argument("--candidates", type=Path, default=root / "reports/m2_pair_candidates.jsonl")
    ap.add_argument("--dimension-audit-csv", type=Path, default=root / "reports/m2_dimension_metadata_audit.csv")
    ap.add_argument("--dimension-audit-json", type=Path, default=root / "reports/m2_dimension_metadata_audit.json")
    ap.add_argument("--analysis-summary", type=Path, default=root / "reports/m2_analysis_summary.json")
    args = ap.parse_args()

    score_maps, result_rows = {}, []
    for model in MODELS:
        rows = read_jsonl(args.score_dir / f"{model}.jsonl")
        score_maps[model] = {row["pair_id"]: row for row in rows}
        for group_type, group_value, subset in groups(rows):
            result_rows.append({"model": model, "group_type": group_type, "group": group_value, **aggregate(subset)})
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0])); writer.writeheader(); writer.writerows(result_rows)

    boot_rows = []
    op = score_maps["openprovence"]; ds = score_maps["ds_modernbert"]
    for group, predicate in [
        ("overall", lambda row: True), ("temporal_all", lambda row: row["dimension"] == "temporal"),
        ("temporal_streamingqa", lambda row: row["dimension"] == "temporal" and row["source"] == "streamingqa"),
        ("temporal_timeqa", lambda row: row["dimension"] == "temporal" and row["source"] == "timeqa"),
    ]:
        ids = [pair_id for pair_id, row in op.items() if pair_id in ds and predicate(row)]
        for metric in ["positive_evidence_recall", "invalid_evidence_retention", "score_gap_mean", "pairwise_discrimination_accuracy"]:
            boot_rows.append({"comparison": "ds_modernbert_minus_openprovence", "group": group, "metric": metric, "iterations": 10000, "seed": SEED, **paired_bootstrap(ds, op, metric, ids)})
    with args.bootstrap.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(boot_rows[0])); writer.writeheader(); writer.writerows(boot_rows)

    case_counts = case_studies(score_maps, args.validity, args.case_md, args.case_html)
    taxonomy = validation_taxonomy(args.supplemental, args.validity, args.taxonomy)
    dimension_audit = dimension_metadata_audit(args.candidates, args.dimension_audit_csv, args.dimension_audit_json)
    collision = json.loads(args.collision_summary.read_text(encoding="utf-8"))
    overall = {row["model"]: row for row in result_rows if row["group_type"] == "overall"}
    temporal = {row["model"]: row for row in result_rows if row["group_type"] == "primary" and row["group"] == "temporal_all"}
    boot_lookup = {(row["group"], row["metric"]): row for row in boot_rows}
    recall_ci = boot_lookup[("overall", "positive_evidence_recall")]
    retention_ci = boot_lookup[("overall", "invalid_evidence_retention")]
    gap_ci = boot_lookup[("overall", "score_gap_mean")]
    discr_ci = boot_lookup[("overall", "pairwise_discrimination_accuracy")]
    recall_maintained = recall_ci["ci95_low"] is not None and recall_ci["ci95_low"] >= -0.03
    retention_improved = retention_ci["ci95_high"] is not None and retention_ci["ci95_high"] < 0
    threshold_free = gap_ci["ci95_low"] is not None and gap_ci["ci95_low"] > 0 and discr_ci["ci95_low"] is not None and discr_ci["ci95_low"] > 0
    if recall_maintained and retention_improved and threshold_free:
        support = "Strongly supported"
    elif retention_improved and (recall_maintained or threshold_free):
        support = "Moderately supported"
    elif overall["ds_modernbert"]["constraint_sensitivity_delta"] > overall["openprovence"]["constraint_sensitivity_delta"]:
        support = "Weak"
    else:
        support = "Contradicted"
    summary = {
        "schema_version": "m2_analysis_summary_v1", "collision_summary": collision, "overall": overall, "temporal": temporal,
        "bootstrap": boot_rows, "case_counts": dict(case_counts), "validation_disagreement_taxonomy": taxonomy,
        "dimension_metadata_audit": dimension_audit,
        "go_no_go": {
            "q1_recall_maintained_and_invalid_retention_reduced": recall_maintained and retention_improved,
            "q2_threshold_independent_difference": threshold_free,
            "q3_temporal_stronger_than_general": temporal["ds_modernbert"]["constraint_sensitivity_delta"] > overall["ds_modernbert"]["constraint_sensitivity_delta"],
            "q4_constraint_sensitivity_support": support,
            "q5_true_hard_negative_precision": collision["hard_negative_validity_rate"],
        },
    }
    args.analysis_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["go_no_go"], indent=2))


if __name__ == "__main__":
    main()
