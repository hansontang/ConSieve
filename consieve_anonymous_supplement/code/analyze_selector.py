#!/usr/bin/env python3
"""Formal three-seed selector analysis for Phase 10.25."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path("${CONSIEVE_ROOT}")
P23 = ROOT / "paper_rewrite/phase10_23"
P25 = ROOT / "paper_rewrite/phase10_25"
ARMS = ("no_cf", "allzero_cf")
SEEDS = (42, 43, 44)
BOOT_SEED = 20260810
BOOT_DRAWS = 10_000
METRICS = (
    "positive_evidence_recall",
    "invalid_evidence_retention",
    "score_gap_mean",
    "pairwise_discrimination_accuracy",
    "auroc",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_m2():
    path = ROOT / "scripts/analyze_m2_stress.py"
    spec = importlib.util.spec_from_file_location("p1025_m2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen M2 analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M2 = load_m2()


def stress_path(arm: str, seed: int) -> Path:
    if seed == 42:
        return P23 / f"results/stress/{arm}.jsonl"
    return P25 / f"results/stress/{arm}_seed{seed}.jsonl"


def gt_path(arm: str, seed: int) -> Path:
    if seed == 42:
        return P23 / f"results/gt406/{arm}_summary.json"
    return P25 / f"results/gt406/{arm}_seed{seed}_summary.json"


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values)}


def metric_table(metrics: dict[str, dict[int, dict[str, Any]]]) -> str:
    lines = [
        "| Arm | Seed | q+ recall | q- invalid retention | Score gap | Pair discr. | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for seed in SEEDS:
            row = metrics[arm][seed]
            lines.append("| " + " | ".join([
                "No-CF-Clean" if arm == "no_cf" else "AllZero-CF-Clean",
                str(seed),
                *(f"{row[key]:.4f}" for key in METRICS),
            ]) + " |")
        lines.append("| " + " | ".join([
            ("No-CF-Clean" if arm == "no_cf" else "AllZero-CF-Clean") + " mean ± SD",
            "—",
            *(f"{mean_sd([metrics[arm][s][key] for s in SEEDS])['mean']:.4f} ± {mean_sd([metrics[arm][s][key] for s in SEEDS])['sd']:.4f}" for key in METRICS),
        ]) + " |")
    return "\n".join(lines)


def hierarchical_bootstrap(rows: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    maps = {
        arm: {seed: {str(row["pair_id"]): row for row in rows[arm][seed]} for seed in SEEDS}
        for arm in ARMS
    }
    ids = sorted(maps["no_cf"][42])
    for arm in ARMS:
        for seed in SEEDS:
            if sorted(maps[arm][seed]) != ids:
                raise RuntimeError(f"paired IDs drift for {arm} seed{seed}")
    point_by_seed: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        point_by_seed[str(seed)] = {
            metric: float(M2.aggregate(rows["allzero_cf"][seed])[metric]) - float(M2.aggregate(rows["no_cf"][seed])[metric])
            for metric in METRICS
        }
    point = {metric: statistics.mean(point_by_seed[str(seed)][metric] for seed in SEEDS) for metric in METRICS}
    rng = random.Random(BOOT_SEED)
    draws = {metric: [] for metric in METRICS}
    for _ in range(BOOT_DRAWS):
        selected_seeds = [SEEDS[rng.randrange(len(SEEDS))] for _ in SEEDS]
        effects = {metric: [] for metric in METRICS}
        for seed in selected_seeds:
            sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
            no_rows = [maps["no_cf"][seed][pair_id] for pair_id in sampled_ids]
            az_rows = [maps["allzero_cf"][seed][pair_id] for pair_id in sampled_ids]
            no_agg, az_agg = M2.aggregate(no_rows), M2.aggregate(az_rows)
            for metric in METRICS:
                effects[metric].append(float(az_agg[metric]) - float(no_agg[metric]))
        for metric in METRICS:
            draws[metric].append(statistics.mean(effects[metric]))
    return {
        "schema_version": "phase10_25_hierarchical_selector_bootstrap/1.0",
        "contrast": "AllZero-CF-Clean minus No-CF-Clean",
        "training_seeds": list(SEEDS),
        "resampling_unit": "diagnostic_pair_id",
        "paired_within_seed": True,
        "bootstrap_draws": BOOT_DRAWS,
        "bootstrap_seed": BOOT_SEED,
        "individual_seed_effects": point_by_seed,
        "metrics": {
            metric: {
                "point_effect": point[metric],
                "ci95_low": pct(draws[metric], 0.025),
                "ci95_high": pct(draws[metric], 0.975),
                "direction": "lower_is_better" if metric == "invalid_evidence_retention" else "higher_is_better",
            }
            for metric in METRICS
        },
    }


def main() -> None:
    P25.mkdir(parents=True, exist_ok=True)
    h96_manifest = json.loads((ROOT / "paper_rewrite/phase10_10/HUMAN_VALID_96_MANIFEST.json").read_text(encoding="utf-8"))
    h96_ids = {str(row["original_diagnostic_id"]) for row in h96_manifest["pairs"]}
    if len(h96_ids) != 96:
        raise RuntimeError("Human-valid-96 identity drift")
    all_rows: dict[str, dict[int, list[dict[str, Any]]]] = {arm: {} for arm in ARMS}
    stress_metrics: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    h96_metrics: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    sources: dict[str, Any] = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = stress_path(arm, seed)
            rows = read_jsonl(path)
            if len(rows) != 412 or len({str(row["pair_id"]) for row in rows}) != 412:
                raise RuntimeError(f"412 denominator drift: {path}")
            subset = [row for row in rows if str(row["pair_id"]) in h96_ids]
            if len(subset) != 96:
                raise RuntimeError(f"H96 coverage drift: {path}")
            all_rows[arm][seed] = rows
            stress_metrics[arm][seed] = M2.aggregate(rows)
            h96_metrics[arm][seed] = M2.aggregate(subset)
            sources[f"{arm}_seed{seed}"] = {"path": str(path), "sha256": sha256(path)}

    bootstrap = hierarchical_bootstrap(all_rows)
    bootstrap["sources"] = sources
    bootstrap_path = P25 / "FORMAL_412_CAUSAL_BOOTSTRAP.json"
    bootstrap_path.write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")
    effect_lines = ["| Metric | AllZero − No-CF | 95% hierarchical paired CI |", "|---|---:|---:|"]
    for metric in METRICS:
        item = bootstrap["metrics"][metric]
        effect_lines.append(f"| {metric} | {item['point_effect']:.4f} | [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] |")
    report = "\n".join([
        "# Formal 412 Three-Seed Causal Selector Results",
        "",
        "Formal arms are exactly No-CF-Clean and AllZero-CF-Clean. Seed42 artifacts are reused from Phase 10.23; seeds43/44 are newly trained under matched settings.",
        "",
        metric_table(stress_metrics),
        "",
        "## Prespecified hierarchical paired bootstrap",
        "",
        *effect_lines,
        "",
        "Signs are interpreted exactly as preregistered: lower invalid retention is better; all other listed effects favor AllZero when positive.",
    ]) + "\n"
    (P25 / "FORMAL_412_CAUSAL_RESULTS.md").write_text(report, encoding="utf-8")

    h96_report = "\n".join([
        "# Formal Human-valid-96 Sensitivity Results",
        "",
        "This is a frozen sensitivity subset; no new annotation was performed.",
        "",
        metric_table(h96_metrics),
    ]) + "\n"
    (P25 / "FORMAL_HUMAN96_RESULTS.md").write_text(h96_report, encoding="utf-8")

    gt: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            path = gt_path(arm, seed)
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("samples") != 406:
                raise RuntimeError(f"GT406 denominator drift: {path}")
            gt[arm][seed] = item
    keys = ("evidence_precision", "evidence_recall", "evidence_f1", "mean_retained_qwen_tokens")
    lines = ["| Arm | Seed | Evidence P | Evidence R | Evidence F1 | Mean retained tokens |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for seed in SEEDS:
            item = gt[arm][seed]
            lines.append(f"| {'No-CF-Clean' if arm == 'no_cf' else 'AllZero-CF-Clean'} | {seed} | {item['evidence_precision']:.4f} | {item['evidence_recall']:.4f} | {item['evidence_f1']:.4f} | {item['mean_retained_qwen_tokens']:.2f} |")
        stats = [mean_sd([float(gt[arm][seed][key]) for seed in SEEDS]) for key in keys]
        lines.append("| " + " | ".join([
            ("No-CF-Clean" if arm == "no_cf" else "AllZero-CF-Clean") + " mean ± SD", "—",
            *(f"{item['mean']:.4f} ± {item['sd']:.4f}" for item in stats[:3]),
            f"{stats[3]['mean']:.2f} ± {stats[3]['sd']:.2f}",
        ]) + " |")
    (P25 / "FORMAL_GT406_SELECTOR_RESULTS.md").write_text("# Formal GT406 Selector Results\n\n" + "\n".join(lines) + "\n\nGT406 is selector-only evidence evaluation and is not used to infer MAIN1500 QA.\n", encoding="utf-8")
    summary = {"stress412": stress_metrics, "human96": h96_metrics, "gt406": gt, "bootstrap_sha256": sha256(bootstrap_path)}
    (P25 / "results/formal_selector_metrics.json").parent.mkdir(parents=True, exist_ok=True)
    (P25 / "results/formal_selector_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "bootstrap": str(bootstrap_path)}, indent=2))


if __name__ == "__main__":
    main()
