#!/usr/bin/env python3
"""Analyze the frozen Phase-10.25 SQuAD confirmatory score artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

ROOT = Path("${CONSIEVE_ROOT}")
P25 = ROOT / "paper_rewrite/phase10_25"
METHODS = ["consieve", "openprovence", "no_cf_seed42", "no_cf_seed43", "no_cf_seed44", "allzero_cf_seed42", "allzero_cf_seed43", "allzero_cf_seed44"]
DISPLAY = {
    "consieve": "Historical canonical ConSieve",
    "openprovence": "OpenProvence",
    "no_cf_seed42": "No-CF-Clean-42",
    "no_cf_seed43": "No-CF-Clean-43",
    "no_cf_seed44": "No-CF-Clean-44",
    "allzero_cf_seed42": "AllZero-CF-Clean-42",
    "allzero_cf_seed43": "AllZero-CF-Clean-43",
    "allzero_cf_seed44": "AllZero-CF-Clean-44",
}
FAMILIES = ["ALL", "ENTITY_SWAP", "NEGATION", "ANTONYM", "NUMBER_SWAP_NON_TEMPORAL"]
SECONDARY_FAMILY = "MUTUAL_EXCLUSION_SECONDARY"
SEEDS = (42, 43, 44)
BOOT_SEED = 20260810
BOOT_DRAWS = 10_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def auroc(pos: list[float], neg: list[float]) -> float:
    scores, labels = pos + neg, [1] * len(pos) + [0] * len(neg)
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for j in range(cursor, end):
            ranks[order[j]] = rank
        cursor = end
    npos, nneg = len(pos), len(neg)
    return (sum(ranks[i] for i, label in enumerate(labels) if label) - npos * (npos + 1) / 2) / (npos * nneg)


def subset(rows, family):
    primary = [row for row in rows if row["analysis_role"] == "PRIMARY"]
    return primary if family == "ALL" else [row for row in primary if row["cause_family"] == family]


def secondary_subset(rows):
    return [row for row in rows if row["analysis_role"] == "SECONDARY" and row["cause_family"] == "MUTUAL_EXCLUSION"]


def metrics(rows, threshold):
    plus = [float(row["qplus_score_full_precision"]) for row in rows]
    minus = [float(row["qminus_score_full_precision"]) for row in rows]
    return {
        "n_pairs": len(rows),
        "n_contexts": len({str(row["context_id"]) for row in rows}),
        "qplus_recall": statistics.mean(float(score > threshold) for score in plus),
        "invalid_retention": statistics.mean(float(score > threshold) for score in minus),
        "auroc": auroc(plus, minus),
        "strict_pair_discrimination": statistics.mean(float(a > b) for a, b in zip(plus, minus)),
        "ties": sum(a == b for a, b in zip(plus, minus)),
    }


def roc_curve(rows):
    plus = [float(row["qplus_score_full_precision"]) for row in rows]
    minus = [float(row["qminus_score_full_precision"]) for row in rows]
    values = plus + minus
    thresholds = [math.nextafter(max(values), math.inf)] + sorted(set(values), reverse=True) + [math.nextafter(min(values), -math.inf)]
    return [
        {
            "threshold": value,
            "valid_recall": statistics.mean(float(score > value) for score in plus),
            "invalid_retention": statistics.mean(float(score > value) for score in minus),
        }
        for value in thresholds
    ]


def canonical_bootstrap(rows_by_method, thresholds):
    output = {}
    for family in FAMILIES:
        left = subset(rows_by_method["openprovence"], family)
        right = subset(rows_by_method["consieve"], family)
        left_by_context, right_by_context = {}, {}
        for row in left:
            left_by_context.setdefault(str(row["context_id"]), []).append(row)
        for row in right:
            right_by_context.setdefault(str(row["context_id"]), []).append(row)
        contexts = sorted(left_by_context)
        if contexts != sorted(right_by_context):
            raise RuntimeError(f"canonical context mismatch: {family}")
        def ir(pool, tau):
            return statistics.mean(float(float(row["qminus_score_full_precision"]) > tau) for row in pool)
        left_stats = {context: (sum(float(float(row["qminus_score_full_precision"]) > thresholds["openprovence"]) for row in pool), len(pool)) for context, pool in left_by_context.items()}
        right_stats = {context: (sum(float(float(row["qminus_score_full_precision"]) > thresholds["consieve"]) for row in pool), len(pool)) for context, pool in right_by_context.items()}
        point = ir(left, thresholds["openprovence"]) - ir(right, thresholds["consieve"])
        rng = random.Random(BOOT_SEED + int(hashlib.sha256(family.encode()).hexdigest()[:8], 16))
        draws = []
        for _ in range(BOOT_DRAWS):
            sampled = [contexts[rng.randrange(len(contexts))] for _ in contexts]
            left_kept, left_n = map(sum, zip(*(left_stats[context] for context in sampled)))
            right_kept, right_n = map(sum, zip(*(right_stats[context] for context in sampled)))
            draws.append(left_kept / left_n - right_kept / right_n)
        output[family] = {
            "contrast": "OpenProvence invalid retention minus canonical ConSieve invalid retention",
            "point_effect": point,
            "ci95_low": percentile(draws, 0.025),
            "ci95_high": percentile(draws, 0.975),
            "contexts": len(contexts),
        }
    return output


def clean_hierarchical(rows_by_method, thresholds):
    output = {}
    individual = {}
    for family in FAMILIES:
        contexts = None
        maps = {}
        for arm in ("no_cf", "allzero_cf"):
            maps[arm] = {}
            for seed in SEEDS:
                key = f"{arm}_seed{seed}"
                rows = subset(rows_by_method[key], family)
                maps[arm][seed] = {}
                for row in rows:
                    maps[arm][seed].setdefault(str(row["context_id"]), []).append(row)
                current = sorted(maps[arm][seed])
                if contexts is None:
                    contexts = current
                elif contexts != current:
                    raise RuntimeError(f"clean context mismatch: {family}/{key}")
        assert contexts is not None
        def invalid_retention(pool, tau):
            return statistics.mean(float(float(row["qminus_score_full_precision"]) > tau) for row in pool)
        context_stats = {
            arm: {
                seed: {
                    context: (
                        sum(float(float(row["qminus_score_full_precision"]) > thresholds[f"{arm}_seed{seed}"]) for row in pool),
                        len(pool),
                    )
                    for context, pool in maps[arm][seed].items()
                }
                for seed in SEEDS
            }
            for arm in ("no_cf", "allzero_cf")
        }
        individual[family] = {}
        for seed in SEEDS:
            no_rows = [row for values in maps["no_cf"][seed].values() for row in values]
            az_rows = [row for values in maps["allzero_cf"][seed].values() for row in values]
            no_ir = invalid_retention(no_rows, thresholds[f"no_cf_seed{seed}"])
            az_ir = invalid_retention(az_rows, thresholds[f"allzero_cf_seed{seed}"])
            individual[family][str(seed)] = no_ir - az_ir
        point = statistics.mean(individual[family].values())
        rng = random.Random(BOOT_SEED + int(hashlib.sha256(("clean:" + family).encode()).hexdigest()[:8], 16))
        draws = []
        for _ in range(BOOT_DRAWS):
            sampled_seeds = [SEEDS[rng.randrange(len(SEEDS))] for _ in SEEDS]
            effects = []
            for seed in sampled_seeds:
                sampled_contexts = [contexts[rng.randrange(len(contexts))] for _ in contexts]
                no_kept, no_n = map(sum, zip(*(context_stats["no_cf"][seed][context] for context in sampled_contexts)))
                az_kept, az_n = map(sum, zip(*(context_stats["allzero_cf"][seed][context] for context in sampled_contexts)))
                no_ir = no_kept / no_n
                az_ir = az_kept / az_n
                effects.append(no_ir - az_ir)
            draws.append(statistics.mean(effects))
        output[family] = {
            "contrast": "No-CF invalid retention minus AllZero-CF invalid retention",
            "point_effect": point,
            "ci95_low": percentile(draws, 0.025),
            "ci95_high": percentile(draws, 0.975),
            "contexts": len(contexts),
            "individual_seed_effects": individual[family],
        }
    return output


def main():
    calibration = json.loads((P25 / "SQUAD_CONFIRM_CALIBRATION_REGISTRY.json").read_text(encoding="utf-8"))
    if calibration.get("status") != "FROZEN":
        raise RuntimeError("calibration registry is not frozen")
    thresholds = {key: float(value["threshold_tau"]) for key, value in calibration["methods"].items()}
    rows_by_method = {}
    for method in METHODS:
        path = P25 / f"results/squad_confirm/{method}_test.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 3405 or len({str(row["pair_id"]) for row in rows}) != 3405:
            raise RuntimeError(f"TEST completeness drift: {method}")
        rows_by_method[method] = rows
    result = {method: {**{family: metrics(subset(rows, family), thresholds[method]) for family in FAMILIES}, SECONDARY_FAMILY: metrics(secondary_subset(rows), thresholds[method])} for method, rows in rows_by_method.items()}
    curves = {method: {**{family: roc_curve(subset(rows, family)) for family in FAMILIES}, SECONDARY_FAMILY: roc_curve(secondary_subset(rows))} for method, rows in rows_by_method.items()}
    curve_path = P25 / "results/SQUAD_CONFIRM_ROC_CURVES.json"
    curve_path.write_text(json.dumps(curves, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    canonical = canonical_bootstrap(rows_by_method, thresholds)
    clean = clean_hierarchical(rows_by_method, thresholds)
    bootstrap = {
        "schema_version": "phase10_25_squad_confirm_bootstrap/1.0",
        "resampling_unit": "context_id_official_paragraph",
        "draws": BOOT_DRAWS,
        "seed": BOOT_SEED,
        "canonical": canonical,
        "clean_hierarchical": clean,
        "roc_curves": str(curve_path),
        "roc_curves_sha256": sha256(curve_path),
    }
    (P25 / "SQUAD_CONFIRM_BOOTSTRAP.json").write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")

    all_rows = ["| Method | Test q+ recall | Invalid retention | AUROC | Pair discr. |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        item = result[method]["ALL"]
        all_rows.append(f"| {DISPLAY[method]} | {item['qplus_recall']:.4f} | {item['invalid_retention']:.4f} | {item['auroc']:.4f} | {item['strict_pair_discrimination']:.4f} |")
    c = canonical["ALL"]
    primary = "\n".join([
        "# SQuAD Confirmatory Primary Results", "",
        "Thresholds were selected independently on the sealed calibration split using the frozen strict IR@95R rule before TEST aggregation.", "",
        *all_rows, "",
        f"Primary canonical contrast (OpenProvence IR − ConSieve IR): **{c['point_effect']:.4f}**, context-clustered 95% CI **[{c['ci95_low']:.4f}, {c['ci95_high']:.4f}]**.", "",
        f"Full valid-recall/invalid-retention curves: `{curve_path}`.",
    ]) + "\n"
    (P25 / "SQUAD_CONFIRM_PRIMARY_RESULTS.md").write_text(primary, encoding="utf-8")

    family_lines = ["# SQuAD Confirmatory Family Results", ""]
    for family in FAMILIES[1:]:
        family_lines += [f"## {family}", "", "| Method | N | q+ recall | Invalid retention | AUROC | Pair discr. |", "|---|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            item = result[method][family]
            family_lines.append(f"| {DISPLAY[method]} | {item['n_pairs']} | {item['qplus_recall']:.4f} | {item['invalid_retention']:.4f} | {item['auroc']:.4f} | {item['strict_pair_discrimination']:.4f} |")
        item = canonical[family]
        family_lines += ["", f"OpenProvence IR − ConSieve IR: {item['point_effect']:.4f} [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}].", ""]
    family_lines += ["## MUTUAL_EXCLUSION (secondary only)", "", "| Method | N | q+ recall | Invalid retention | AUROC | Pair discr. |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        item = result[method][SECONDARY_FAMILY]
        family_lines.append(f"| {DISPLAY[method]} | {item['n_pairs']} | {item['qplus_recall']:.4f} | {item['invalid_retention']:.4f} | {item['auroc']:.4f} | {item['strict_pair_discrimination']:.4f} |")
    family_lines += ["", "This 90-pair family remains secondary exactly as frozen and is not used in the primary result gate.", ""]
    (P25 / "SQUAD_CONFIRM_FAMILY_RESULTS.md").write_text("\n".join(family_lines) + "\n", encoding="utf-8")

    con, op = result["consieve"]["ALL"], result["openprovence"]["ALL"]
    conditions = {
        "reasonable_test_qplus_recall_preferably_ge_0_93": con["qplus_recall"] >= .93 and op["qplus_recall"] >= .93,
        "consieve_lower_invalid_retention": con["invalid_retention"] < op["invalid_retention"],
        "canonical_context_ci_excludes_zero_positive": c["ci95_low"] > 0,
        "auroc_directionally_consistent": con["auroc"] > op["auroc"],
        "no_primary_family_directional_reversal_in_invalid_retention": all(result["consieve"][family]["invalid_retention"] <= result["openprovence"][family]["invalid_retention"] for family in FAMILIES[1:]),
    }
    classification = "STRONG SUPPORT" if all(conditions.values()) else ("MIXED" if conditions["consieve_lower_invalid_retention"] else ("NULL" if abs(c["point_effect"]) < .01 else "NEGATIVE"))
    classification_payload = {"classification": classification, "frozen_gate_conditions": conditions, "canonical_metrics": {"consieve": con, "openprovence": op}, "canonical_contrast": c}
    (P25 / "results/squad_confirm_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (P25 / "SQUAD_CONFIRM_RESULT_CLASSIFICATION.md").write_text("# SQuAD Confirmatory Result Classification\n\nClassification: **" + classification + "**.\n\n" + "\n".join(f"- {key}: **{'PASS' if value else 'FAIL'}**" for key, value in conditions.items()) + "\n\nThe bounded claim concerns human-written non-temporal invalid-evidence rejection at calibration-matched valid recall; arbitrary constraint generalization remains unsupported.\n", encoding="utf-8")
    print(json.dumps(classification_payload, indent=2))


if __name__ == "__main__":
    main()
