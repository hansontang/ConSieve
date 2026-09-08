#!/usr/bin/env python3
"""Compute the blinded two-annotator M2 human-audit statistics.

Inputs are normalized JSON exported from the source XLSX workbooks with
artifact-tool plus the frozen admin manifest.  No adjudication is inferred.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


LABELS = ("SUPPORTS", "DOES_NOT_SUPPORT", "AMBIGUOUS")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float | None]:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def cohen_kappa(xs: list[str], ys: list[str], labels: tuple[str, ...]) -> float:
    if len(xs) != len(ys) or not xs:
        raise ValueError("Kappa inputs must be non-empty and aligned")
    n = len(xs)
    po = sum(a == b for a, b in zip(xs, ys)) / n
    cx, cy = Counter(xs), Counter(ys)
    pe = sum((cx[label] / n) * (cy[label] / n) for label in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def evidence_set(value: object) -> set[str]:
    if value is None:
        return set()
    return set(re.findall(r"S\d{3}", str(value).upper()))


def normalize_answer(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def answer_f1(a: object, b: object) -> float:
    ta, tb = normalize_answer(a), normalize_answer(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    common = sum((ca & cb).values())
    if common == 0:
        return 0.0
    precision = common / len(ta)
    recall = common / len(tb)
    return 2 * precision * recall / (precision + recall)


def pair_pattern(row: dict) -> str:
    a, b = row["A_Label"], row["B_Label"]
    if "AMBIGUOUS" in (a, b):
        return "AMBIGUOUS"
    if a == "SUPPORTS" and b == "DOES_NOT_SUPPORT":
        return "A_ONLY"
    if a == "DOES_NOT_SUPPORT" and b == "SUPPORTS":
        return "B_ONLY"
    if a == "SUPPORTS" and b == "SUPPORTS":
        return "BOTH"
    if a == "DOES_NOT_SUPPORT" and b == "DOES_NOT_SUPPORT":
        return "NEITHER"
    raise ValueError(f"Unexpected labels: {a}, {b}")


def annotate_pair(row: dict, admin: dict) -> dict:
    original_side = admin["original_as"]
    perturbed_side = "B" if original_side == "A" else "A"
    original_label = row[f"{original_side}_Label"]
    perturbed_label = row[f"{perturbed_side}_Label"]
    return {
        "pattern": pair_pattern(row),
        "original_label": original_label,
        "perturbed_label": perturbed_label,
        "original_support": original_label == "SUPPORTS",
        "perturbed_reject": perturbed_label == "DOES_NOT_SUPPORT",
        "strict_valid": original_label == "SUPPORTS" and perturbed_label == "DOES_NOT_SUPPORT",
    }


def rate_record(k: int, n: int) -> dict:
    return {"count": k, "n": n, "rate": k / n if n else None, "wilson_95": wilson(k, n)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--admin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-workbook", type=Path, action="append", default=[])
    parser.add_argument("--adjudication", type=Path)
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    admin = json.loads(args.admin.read_text(encoding="utf-8"))
    names = list(annotations)
    if len(names) != 2:
        raise ValueError(f"Expected exactly two annotators, found {names}")
    name_a, name_b = names
    public = {name_a: "Annotator_A", name_b: "Annotator_B"}
    rows_by_name = {
        name: {row["Audit_ID"]: row for row in annotations[name]["rows"] if row.get("Audit_ID")}
        for name in names
    }
    admin_by_id = {row["audit_id"]: row for row in admin["samples"]}
    ids = sorted(admin_by_id)
    for name in names:
        if set(rows_by_name[name]) != set(ids):
            raise ValueError(f"Audit-ID mismatch for {name}")

    pair_rows: list[dict] = []
    flat_a: list[str] = []
    flat_b: list[str] = []
    per_annotator: dict[str, dict] = {}
    derived_by_name: dict[str, dict[str, dict]] = {}

    for name in names:
        derived_by_name[name] = {}
        strict = original = perturbed = 0
        patterns = Counter()
        label_counts = Counter()
        confidence = Counter()
        note_count = 0
        reviewer_id_count = 0
        date_count = 0
        support_missing_required = 0
        nonsupport_with_extra = 0
        invalid_evidence_references = []
        malformed_evidence_fields = []
        for audit_id in ids:
            row = rows_by_name[name][audit_id]
            d = annotate_pair(row, admin_by_id[audit_id])
            derived_by_name[name][audit_id] = d
            strict += d["strict_valid"]
            original += d["original_support"]
            perturbed += d["perturbed_reject"]
            patterns[d["pattern"]] += 1
            label_counts.update([row["A_Label"], row["B_Label"]])
            confidence[str(row.get("Confidence") or "MISSING")] += 1
            note_count += bool(str(row.get("Notes") or "").strip())
            reviewer_id_count += bool(str(row.get("Annotator_ID") or "").strip())
            date_count += bool(str(row.get("Annotated_Date") or "").strip())
            for side in ("A", "B"):
                label = row[f"{side}_Label"]
                evidence_raw = row.get(f"{side}_Evidence_IDs")
                answer_raw = row.get(f"{side}_Short_Answer")
                evidence = evidence_set(evidence_raw)
                if label == "SUPPORTS" and (not evidence or not str(answer_raw or "").strip()):
                    support_missing_required += 1
                if str(evidence_raw or "").strip() and not evidence:
                    malformed_evidence_fields.append({"audit_id": audit_id, "side": side, "value": str(evidence_raw)})
                if label != "SUPPORTS" and (str(evidence_raw or "").strip() or str(answer_raw or "").strip()):
                    nonsupport_with_extra += 1
                max_sentence = admin_by_id[audit_id]["sentence_count"]
                bad = sorted(s for s in evidence if int(s[1:]) < 1 or int(s[1:]) > max_sentence)
                if bad:
                    invalid_evidence_references.append({"audit_id": audit_id, "side": side, "ids": bad})
        per_annotator[public[name]] = {
            "pairs": len(ids),
            "side_decisions": 2 * len(ids),
            "label_counts": dict(label_counts),
            "pair_patterns": dict(patterns),
            "original_query_support": rate_record(original, len(ids)),
            "perturbed_query_rejection": rate_record(perturbed, len(ids)),
            "strict_pair_validity": rate_record(strict, len(ids)),
            "confidence_counts": dict(confidence),
            "notes_nonempty": note_count,
            "reviewer_id_nonempty": reviewer_id_count,
            "annotation_date_nonempty": date_count,
            "content_completion": {
                "support_decisions_missing_evidence_or_answer": support_missing_required,
                "nonsupport_decisions_with_evidence_or_answer": nonsupport_with_extra,
                "invalid_evidence_references": invalid_evidence_references,
                "malformed_evidence_fields": malformed_evidence_fields,
            },
        }

    for audit_id in ids:
        ra, rb = rows_by_name[name_a][audit_id], rows_by_name[name_b][audit_id]
        da, db = derived_by_name[name_a][audit_id], derived_by_name[name_b][audit_id]
        admin_row = admin_by_id[audit_id]
        for side in ("A", "B"):
            flat_a.append(ra[f"{side}_Label"])
            flat_b.append(rb[f"{side}_Label"])
        pair_rows.append({
            "audit_id": audit_id,
            "pair_id": admin_row["pair_id"],
            "source": admin_row["source"],
            "original_side": admin_row["original_as"],
            "question_a": admin_row["question_a"],
            "question_b": admin_row["question_b"],
            "annotator_a_a_label": ra["A_Label"],
            "annotator_a_b_label": ra["B_Label"],
            "annotator_b_a_label": rb["A_Label"],
            "annotator_b_b_label": rb["B_Label"],
            "annotator_a_pattern": da["pattern"],
            "annotator_b_pattern": db["pattern"],
            "annotator_a_strict_valid": da["strict_valid"],
            "annotator_b_strict_valid": db["strict_valid"],
            "annotator_a_confidence": ra.get("Confidence"),
            "annotator_b_confidence": rb.get("Confidence"),
            "annotator_a_notes": ra.get("Notes") or "",
            "annotator_b_notes": rb.get("Notes") or "",
            "side_label_agreement": int(ra["A_Label"] == rb["A_Label"]) + int(ra["B_Label"] == rb["B_Label"]),
            "exact_pair_label_agreement": ra["A_Label"] == rb["A_Label"] and ra["B_Label"] == rb["B_Label"],
            "strict_validity_agreement": da["strict_valid"] == db["strict_valid"],
        })

    exact_side = sum(a == b for a, b in zip(flat_a, flat_b))
    nonamb_idx = [i for i, (a, b) in enumerate(zip(flat_a, flat_b)) if "AMBIGUOUS" not in (a, b)]
    bin_a = ["SUPPORTS" if flat_a[i] == "SUPPORTS" else "DOES_NOT_SUPPORT" for i in nonamb_idx]
    bin_b = ["SUPPORTS" if flat_b[i] == "SUPPORTS" else "DOES_NOT_SUPPORT" for i in nonamb_idx]
    exact_pairs = sum(row["exact_pair_label_agreement"] for row in pair_rows)
    strict_agree = sum(row["strict_validity_agreement"] for row in pair_rows)
    both_valid = sum(row["annotator_a_strict_valid"] and row["annotator_b_strict_valid"] for row in pair_rows)
    both_not_valid = sum((not row["annotator_a_strict_valid"]) and (not row["annotator_b_strict_valid"]) for row in pair_rows)
    one_valid = len(pair_rows) - both_valid - both_not_valid
    exact_resolved = [row for row in pair_rows if row["exact_pair_label_agreement"]]
    exact_valid = sum(row["annotator_a_strict_valid"] for row in exact_resolved)

    by_source: dict[str, dict] = {}
    for source in sorted({row["source"] for row in pair_rows}):
        source_rows = [row for row in pair_rows if row["source"] == source]
        source_exact = [row for row in source_rows if row["exact_pair_label_agreement"]]
        by_source[source] = {
            "n": len(source_rows),
            "exact_pair_label_agreement": rate_record(len(source_exact), len(source_rows)),
            "annotator_a_strict_validity": rate_record(sum(row["annotator_a_strict_valid"] for row in source_rows), len(source_rows)),
            "annotator_b_strict_validity": rate_record(sum(row["annotator_b_strict_valid"] for row in source_rows), len(source_rows)),
            "exact_resolved_strict_validity": rate_record(sum(row["annotator_a_strict_valid"] for row in source_exact), len(source_exact)),
        }

    by_side = {}
    for side in ("A", "B"):
        xs = [rows_by_name[name_a][audit_id][f"{side}_Label"] for audit_id in ids]
        ys = [rows_by_name[name_b][audit_id][f"{side}_Label"] for audit_id in ids]
        by_side[side] = {
            "agreement": rate_record(sum(a == b for a, b in zip(xs, ys)), len(xs)),
            "cohen_kappa_3class": cohen_kappa(xs, ys, LABELS),
        }

    mutual_support = []
    for audit_id in ids:
        ra, rb = rows_by_name[name_a][audit_id], rows_by_name[name_b][audit_id]
        for side in ("A", "B"):
            if ra[f"{side}_Label"] == rb[f"{side}_Label"] == "SUPPORTS":
                ea = evidence_set(ra.get(f"{side}_Evidence_IDs"))
                eb = evidence_set(rb.get(f"{side}_Evidence_IDs"))
                union = ea | eb
                mutual_support.append({
                    "audit_id": audit_id,
                    "side": side,
                    "evidence_exact": ea == eb,
                    "evidence_jaccard": len(ea & eb) / len(union) if union else 1.0,
                    "answer_exact": normalize_answer(ra.get(f"{side}_Short_Answer")) == normalize_answer(rb.get(f"{side}_Short_Answer")),
                    "answer_f1": answer_f1(ra.get(f"{side}_Short_Answer"), rb.get(f"{side}_Short_Answer")),
                })

    disagreements = [row for row in pair_rows if not row["exact_pair_label_agreement"]]
    summary = {
        "schema_version": "m2_human_audit_report_v1",
        "population": {
            "frozen_pairs": admin["frozen_population"],
            "sample_pairs": admin["sample_size"],
            "seed": admin["seed"],
            "sampling_method": admin["sampling_method"],
            "source_counts": admin["source_counts"],
            "original_side_counts": admin["original_side_counts"],
            "population_manifest_sha256": admin["population_manifest_sha256"],
            "ordered_sample_set_sha256": admin["ordered_sample_set_sha256"],
        },
        "annotators": per_annotator,
        "inter_annotator": {
            "side_decisions": len(flat_a),
            "side_label_agreement": rate_record(exact_side, len(flat_a)),
            "cohen_kappa_3class": cohen_kappa(flat_a, flat_b, LABELS),
            "binary_nonambiguous_side_decisions": len(nonamb_idx),
            "binary_nonambiguous_agreement": rate_record(sum(a == b for a, b in zip(bin_a, bin_b)), len(bin_a)),
            "cohen_kappa_binary_nonambiguous": cohen_kappa(bin_a, bin_b, ("SUPPORTS", "DOES_NOT_SUPPORT")),
            "by_side": by_side,
            "exact_pair_label_agreement": rate_record(exact_pairs, len(pair_rows)),
            "strict_validity_agreement": rate_record(strict_agree, len(pair_rows)),
            "strict_validity_cross_tab": {
                "both_valid": both_valid,
                "both_not_valid": both_not_valid,
                "one_valid_one_not": one_valid,
            },
            "exact_resolved_pairs": {
                "n": len(exact_resolved),
                "strict_valid": exact_valid,
                "strict_invalid": len(exact_resolved) - exact_valid,
                "strict_validity_rate": rate_record(exact_valid, len(exact_resolved)),
            },
            "strict_validity_identification_bounds_before_adjudication": {
                "lower": both_valid / len(pair_rows),
                "upper": (both_valid + one_valid) / len(pair_rows),
                "note": "Descriptive bounds only: lower treats all disputed pairs as invalid; upper treats all disputed pairs as valid.",
            },
            "by_source": by_source,
            "mutual_support_evidence": {
                "n": len(mutual_support),
                "exact_set_agreement": sum(x["evidence_exact"] for x in mutual_support) / len(mutual_support) if mutual_support else None,
                "mean_jaccard": sum(x["evidence_jaccard"] for x in mutual_support) / len(mutual_support) if mutual_support else None,
                "short_answer_normalized_exact": sum(x["answer_exact"] for x in mutual_support) / len(mutual_support) if mutual_support else None,
                "short_answer_mean_token_f1": sum(x["answer_f1"] for x in mutual_support) / len(mutual_support) if mutual_support else None,
            },
            "pairs_requiring_label_adjudication": len(disagreements),
        },
        "interpretation_guardrail": "No third-adjudicator file was supplied. Strict-validity rates are reported per annotator; disagreements remain unresolved and are not silently adjudicated.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "m2_human_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    normalized_fields = (
        "Audit_ID", "Pair_ID", "Source", "Question_A", "Question_B",
        "A_Label", "A_Evidence_IDs", "A_Short_Answer",
        "B_Label", "B_Evidence_IDs", "B_Short_Answer",
        "Confidence", "Notes", "Completion_Check", "Annotator_ID", "Annotated_Date",
    )
    normalized = {
        public[name]: {
            "rows": [{field: rows_by_name[name][audit_id].get(field) for field in normalized_fields} for audit_id in ids]
        }
        for name in names
    }
    (args.output_dir / "m2_human_audit_annotations_normalized.json").write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fields = list(pair_rows[0])
    with (args.output_dir / "m2_human_audit_pairwise.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)
    with (args.output_dir / "m2_human_audit_disagreements.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(disagreements)

    adjudication_fields = [
        "Audit_ID", "Pair_ID", "Source", "Question_A", "Question_B",
        "Document_with_Sentence_IDs", "A_Label", "A_Evidence_IDs", "A_Short_Answer",
        "B_Label", "B_Evidence_IDs", "B_Short_Answer", "Confidence", "Notes",
        "Adjudicator_ID", "Adjudicated_Date",
    ]
    with (args.output_dir / "m2_human_audit_adjudication_blinded.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=adjudication_fields)
        writer.writeheader()
        for row in disagreements:
            admin_row = admin_by_id[row["audit_id"]]
            writer.writerow({
                "Audit_ID": row["audit_id"],
                "Pair_ID": row["pair_id"],
                "Source": row["source"],
                "Question_A": row["question_a"],
                "Question_B": row["question_b"],
                "Document_with_Sentence_IDs": admin_row["document"],
                "A_Label": "",
                "A_Evidence_IDs": "",
                "A_Short_Answer": "",
                "B_Label": "",
                "B_Evidence_IDs": "",
                "B_Short_Answer": "",
                "Confidence": "",
                "Notes": "",
                "Adjudicator_ID": "",
                "Adjudicated_Date": "",
            })

    manifest = {
        "source_files": [],
        "normalization": "Source XLSX workbooks were imported with artifact-tool and normalized to JSON before this analysis.",
    }
    for path in (args.annotations, args.admin, *args.source_workbook):
        manifest["source_files"].append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    (args.output_dir / "m2_human_audit_analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.adjudication:
        with args.adjudication.open("r", encoding="utf-8-sig", newline="") as fh:
            adjudication_reader = csv.DictReader(fh)
            adjudication_rows_raw = list(adjudication_reader)
            adjudication_headers = adjudication_reader.fieldnames or []
        if "original_side" in adjudication_headers:
            raise ValueError("Blinded adjudication file unexpectedly exposes original_side")
        adjudication_by_id = {row["Audit_ID"]: row for row in adjudication_rows_raw}
        disagreement_ids = {row["audit_id"] for row in disagreements}
        if set(adjudication_by_id) != disagreement_ids:
            raise ValueError(
                f"Adjudication IDs do not match disagreements: "
                f"missing={sorted(disagreement_ids-set(adjudication_by_id))}, "
                f"extra={sorted(set(adjudication_by_id)-disagreement_ids)}"
            )

        alias_map = {"SUPPORT": "SUPPORTS"}
        normalization_events = []
        adjudication_normalized = {}
        adjudication_content_issues = []
        for audit_id in sorted(adjudication_by_id):
            raw = adjudication_by_id[audit_id]
            admin_row = admin_by_id[audit_id]
            for field, expected in (
                ("Pair_ID", admin_row["pair_id"]),
                ("Source", admin_row["source"]),
                ("Question_A", admin_row["question_a"]),
                ("Question_B", admin_row["question_b"]),
            ):
                if raw.get(field) != expected:
                    raise ValueError(f"Adjudication mismatch at {audit_id} {field}")
            normalized = dict(raw)
            for side in ("A", "B"):
                field = f"{side}_Label"
                raw_label = str(raw.get(field) or "").strip().upper()
                label = alias_map.get(raw_label, raw_label)
                if raw_label != label:
                    normalization_events.append({
                        "audit_id": audit_id,
                        "field": field,
                        "raw": raw_label,
                        "normalized": label,
                        "rule": "documented exact alias SUPPORT -> SUPPORTS",
                    })
                if label not in LABELS:
                    raise ValueError(f"Invalid adjudication label at {audit_id} {field}: {raw_label!r}")
                normalized[field] = label
                ev_raw = raw.get(f"{side}_Evidence_IDs")
                answer_raw = raw.get(f"{side}_Short_Answer")
                ev = evidence_set(ev_raw)
                if label == "SUPPORTS" and (not ev or not str(answer_raw or "").strip()):
                    adjudication_content_issues.append({
                        "audit_id": audit_id, "side": side,
                        "issue": "SUPPORTS missing valid evidence ID or short answer",
                    })
                if label != "SUPPORTS" and (str(ev_raw or "").strip() or str(answer_raw or "").strip()):
                    adjudication_content_issues.append({
                        "audit_id": audit_id, "side": side,
                        "issue": "non-SUPPORTS has evidence or short answer",
                    })
                bad = sorted(s for s in ev if int(s[1:]) < 1 or int(s[1:]) > admin_row["sentence_count"])
                if bad:
                    adjudication_content_issues.append({
                        "audit_id": audit_id, "side": side,
                        "issue": f"out-of-range evidence IDs: {bad}",
                    })
            adjudication_normalized[audit_id] = normalized

        final_rows = []
        adjudicator_pair_valid = 0
        adjudicator_matches_a = 0
        adjudicator_matches_b = 0
        adjudicator_matches_neither = 0
        for row in pair_rows:
            audit_id = row["audit_id"]
            admin_row = admin_by_id[audit_id]
            if row["exact_pair_label_agreement"]:
                source = "CONSENSUS"
                final_a = row["annotator_a_a_label"]
                final_b = row["annotator_a_b_label"]
            else:
                source = "ADJUDICATED"
                adj = adjudication_normalized[audit_id]
                final_a, final_b = adj["A_Label"], adj["B_Label"]
                adjud_derived = annotate_pair(adj, admin_row)
                adjudicator_pair_valid += adjud_derived["strict_valid"]
                matches_a = (final_a, final_b) == (row["annotator_a_a_label"], row["annotator_a_b_label"])
                matches_b = (final_a, final_b) == (row["annotator_b_a_label"], row["annotator_b_b_label"])
                adjudicator_matches_a += matches_a
                adjudicator_matches_b += matches_b
                adjudicator_matches_neither += not matches_a and not matches_b
            final_input = {"A_Label": final_a, "B_Label": final_b}
            final_derived = annotate_pair(final_input, admin_row)
            final_rows.append({
                "audit_id": audit_id,
                "pair_id": row["pair_id"],
                "source": row["source"],
                "original_side": row["original_side"],
                "question_a": row["question_a"],
                "question_b": row["question_b"],
                "final_a_label": final_a,
                "final_b_label": final_b,
                "final_pattern": final_derived["pattern"],
                "original_label": final_derived["original_label"],
                "perturbed_label": final_derived["perturbed_label"],
                "original_support": final_derived["original_support"],
                "perturbed_reject": final_derived["perturbed_reject"],
                "strict_valid": final_derived["strict_valid"],
                "final_label_source": source,
            })

        final_valid = sum(row["strict_valid"] for row in final_rows)
        final_original = sum(row["original_support"] for row in final_rows)
        final_perturbed = sum(row["perturbed_reject"] for row in final_rows)
        final_patterns = Counter(row["final_pattern"] for row in final_rows)
        final_labels = Counter(x for row in final_rows for x in (row["final_a_label"], row["final_b_label"]))
        final_by_source = {}
        for source in sorted({row["source"] for row in final_rows}):
            source_rows = [row for row in final_rows if row["source"] == source]
            final_by_source[source] = {
                "n": len(source_rows),
                "strict_validity": rate_record(sum(row["strict_valid"] for row in source_rows), len(source_rows)),
                "original_query_support": rate_record(sum(row["original_support"] for row in source_rows), len(source_rows)),
                "perturbed_query_rejection": rate_record(sum(row["perturbed_reject"] for row in source_rows), len(source_rows)),
            }

        final_invalid = []
        for row in final_rows:
            if row["strict_valid"]:
                continue
            if not row["original_support"] and row["perturbed_reject"]:
                failure = "original query not supported"
            elif row["original_support"] and not row["perturbed_reject"]:
                failure = "perturbed query also supported (collision)"
            elif not row["original_support"] and not row["perturbed_reject"]:
                failure = "original unsupported and perturbed not rejected"
            else:
                failure = "ambiguous strict-validity failure"
            final_invalid.append({
                "audit_id": row["audit_id"],
                "source": row["source"],
                "pattern": row["final_pattern"],
                "failure_mode": failure,
                "final_label_source": row["final_label_source"],
            })

        final_summary = {
            "schema_version": "m2_human_audit_final_report_v1",
            "status": "FINAL_AFTER_BLINDED_ADJUDICATION",
            "population": summary["population"],
            "initial_inter_annotator": summary["inter_annotator"],
            "adjudication": {
                "required_pairs": len(disagreements),
                "completed_pairs": len(adjudication_normalized),
                "strict_valid": adjudicator_pair_valid,
                "strict_invalid": len(adjudication_normalized) - adjudicator_pair_valid,
                "exact_pair_match_annotator_a": adjudicator_matches_a,
                "exact_pair_match_annotator_b": adjudicator_matches_b,
                "exact_pair_match_neither": adjudicator_matches_neither,
                "confidence_counts": dict(Counter(str(row.get("Confidence") or "MISSING").upper() for row in adjudication_normalized.values())),
                "adjudicator_id_nonempty": sum(bool(str(row.get("Adjudicator_ID") or "").strip()) for row in adjudication_normalized.values()),
                "adjudicated_date_nonempty": sum(bool(str(row.get("Adjudicated_Date") or "").strip()) for row in adjudication_normalized.values()),
                "normalization_events": normalization_events,
                "content_issues": adjudication_content_issues,
            },
            "final": {
                "pairs": len(final_rows),
                "strict_pair_validity": rate_record(final_valid, len(final_rows)),
                "original_query_support": rate_record(final_original, len(final_rows)),
                "perturbed_query_rejection": rate_record(final_perturbed, len(final_rows)),
                "pair_pattern_counts": dict(final_patterns),
                "side_label_counts": dict(final_labels),
                "label_source_counts": dict(Counter(row["final_label_source"] for row in final_rows)),
                "by_source": final_by_source,
                "invalid_pairs": final_invalid,
            },
            "reporting_guardrail": "The final 96/100 estimate applies to the uniform sample from the frozen verified 412 only, not the original 1,646 candidate pairs.",
        }
        (args.output_dir / "m2_human_audit_final_summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with (args.output_dir / "m2_human_audit_final_labels.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(final_rows[0]))
            writer.writeheader()
            writer.writerows(final_rows)
        normalized_adj_fields = [
            "Audit_ID", "Pair_ID", "Source", "Question_A", "Question_B",
            "A_Label", "A_Evidence_IDs", "A_Short_Answer",
            "B_Label", "B_Evidence_IDs", "B_Short_Answer", "Confidence", "Notes",
            "Adjudicator_ID", "Adjudicated_Date",
        ]
        with (args.output_dir / "m2_human_audit_adjudication_normalized.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=normalized_adj_fields)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in normalized_adj_fields} for row in adjudication_normalized.values())
        final_manifest = {
            "analysis_script": str(Path(__file__)),
            "source_files": [],
            "label_normalization_events": normalization_events,
            "frozen_artifacts_modified": False,
        }
        for path in (args.annotations, args.admin, args.adjudication, *args.source_workbook):
            final_manifest["source_files"].append({
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        (args.output_dir / "m2_human_audit_final_manifest.json").write_text(
            json.dumps(final_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
