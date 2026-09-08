#!/usr/bin/env python3
"""Formal three-seed MAIN1500 clean-arm QA analysis."""
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

import numpy as np

ROOT = Path("${CONSIEVE_ROOT}")
P25 = ROOT / "paper_rewrite/phase10_25"
ARMS = ("no_cf", "allzero_cf")
SEEDS = (42, 43, 44)
BUDGETS = (128, 256, 512)
METRICS = ("f1", "em", "judge")
BOOT_DRAWS = 10_000
BOOT_SEED = 20260810

spec = importlib.util.spec_from_file_location("p1025_m5_metrics", ROOT / "scripts/analyze_m5_results.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import frozen M5 metrics")
M5 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M5
spec.loader.exec_module(M5)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def output_path(arm: str, seed: int, budget: int) -> Path:
    return P25 / f"results/main1500_qa/{arm}_seed{seed}_budget{budget}.json"


def validate(rows: list[dict[str, Any]], path: Path, budget: int):
    manifest = json.loads((ROOT / "manifests/main1500.json").read_text(encoding="utf-8"))
    expected = [str(row["id"]) for row in manifest["samples"]]
    actual = [str(row["id"]) for row in rows]
    if actual != expected or len(set(actual)) != 1500:
        raise RuntimeError(f"MAIN1500 ID/order drift: {path}")
    if any(row.get("generation_status") != "success" or row.get("judge_status") != "judged" for row in rows):
        raise RuntimeError(f"unresolved API failure: {path}")
    if any(int(row["context_tokens"]) > budget for row in rows):
        raise RuntimeError(f"budget violation: {path}")
    audit = json.loads(path.with_suffix(".audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("generation_success") != 1500 or audit.get("judge_success") != 1500 or audit.get("unrecovered_provider_errors") != 0:
        raise RuntimeError(f"QA audit incomplete: {path}")


def per_values(rows):
    return {
        "f1": [float(M5.per_f1(row)) for row in rows],
        "em": [float(M5.per_em(row)) for row in rows],
        "judge": [float(bool(row.get("judge_correct"))) for row in rows],
    }


def summarize(rows, values):
    return {
        "n": len(rows),
        "f1": statistics.mean(values["f1"]),
        "em": statistics.mean(values["em"]),
        "judge": statistics.mean(values["judge"]),
        "mean_context_tokens": statistics.mean(int(row["context_tokens"]) for row in rows),
        "empty_context_rate": statistics.mean(float(not str(row.get("context", "")).strip()) for row in rows),
    }


def percentile(values: list[float], q: float):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def hierarchical(values, budget):
    differences = {
        seed: {
            metric: np.asarray(values["allzero_cf"][seed][budget][metric], dtype=np.float64)
            - np.asarray(values["no_cf"][seed][budget][metric], dtype=np.float64)
            for metric in METRICS
        }
        for seed in SEEDS
    }
    seed_effects = {
        str(seed): {
            metric: float(differences[seed][metric].mean())
            for metric in METRICS
        }
        for seed in SEEDS
    }
    point = {metric: statistics.mean(seed_effects[str(seed)][metric] for seed in SEEDS) for metric in METRICS}
    rng = np.random.default_rng(BOOT_SEED + budget)
    draws = {metric: [] for metric in METRICS}
    n = 1500
    for _ in range(BOOT_DRAWS):
        sampled_seeds = [SEEDS[index] for index in rng.integers(0, len(SEEDS), size=len(SEEDS))]
        effects = {metric: [] for metric in METRICS}
        for seed in sampled_seeds:
            indices = rng.integers(0, n, size=n)
            for metric in METRICS:
                effects[metric].append(float(differences[seed][metric][indices].mean()))
        for metric in METRICS:
            draws[metric].append(statistics.mean(effects[metric]))
    return {
        "contrast": "AllZero-CF-Clean minus No-CF-Clean",
        "budget": budget,
        "training_seeds": list(SEEDS),
        "query_resampling_unit": "MAIN1500 query ID",
        "paired_within_seed": True,
        "draws": BOOT_DRAWS,
        "seed": BOOT_SEED,
        "individual_seed_effects": seed_effects,
        "metrics": {
            metric: {"point_effect": point[metric], "ci95_low": percentile(draws[metric], .025), "ci95_high": percentile(draws[metric], .975)}
            for metric in METRICS
        },
    }


def main():
    freeze = json.loads((P25 / "FORMAL_METHOD_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("formal method freeze missing")
    rows = {arm: {seed: {} for seed in SEEDS} for arm in ARMS}
    values = {arm: {seed: {} for seed in SEEDS} for arm in ARMS}
    summaries = {arm: {seed: {} for seed in SEEDS} for arm in ARMS}
    artifacts = {}
    for arm in ARMS:
        for seed in SEEDS:
            for budget in BUDGETS:
                path = output_path(arm, seed, budget)
                current = json.loads(path.read_text(encoding="utf-8"))
                validate(current, path, budget)
                rows[arm][seed][budget] = current
                values[arm][seed][budget] = per_values(current)
                summaries[arm][seed][budget] = summarize(current, values[arm][seed][budget])
                artifacts[f"{arm}_seed{seed}_budget{budget}"] = {"path": str(path), "sha256": sha256(path), "audit_sha256": sha256(path.with_suffix(".audit.json"))}
    bootstraps = {str(budget): hierarchical(values, budget) for budget in BUDGETS}
    payload = {
        "schema_version": "phase10_25_main1500_hierarchical_bootstrap/1.0",
        "primary_budget": 256,
        "budgets": list(BUDGETS),
        "bootstrap": bootstraps,
        "artifacts": artifacts,
    }
    (P25 / "MAIN1500_CLEAN_CAUSAL_BOOTSTRAP.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (P25 / "results/main1500_clean_metrics.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    lines = ["| Arm | Seed | Budget | F1 | EM | Judge | Mean tokens | Empty |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for seed in SEEDS:
            for budget in BUDGETS:
                item = summaries[arm][seed][budget]
                lines.append(f"| {'No-CF-Clean' if arm == 'no_cf' else 'AllZero-CF-Clean'} | {seed} | {budget} | {item['f1']:.4f} | {item['em']:.4f} | {item['judge']:.4f} | {item['mean_context_tokens']:.2f} | {item['empty_context_rate']:.4f} |")
    primary = bootstraps["256"]["metrics"]
    primary_lines = [f"- {metric.upper()}: {item['point_effect']:.4f} [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}]" for metric, item in primary.items()]
    (P25 / "MAIN1500_CLEAN_CAUSAL_RESULTS.md").write_text("# MAIN1500 Formal Clean Causal QA Results\n\n" + "\n".join(lines) + "\n\n## Primary 256-token hierarchical causal contrast\n\n" + "\n".join(primary_lines) + "\n", encoding="utf-8")

    curve = ["| Budget | F1 effect | 95% CI | EM effect | Judge effect |", "|---:|---:|---:|---:|---:|"]
    for budget in BUDGETS:
        item = bootstraps[str(budget)]["metrics"]
        curve.append(f"| {budget} | {item['f1']['point_effect']:.4f} | [{item['f1']['ci95_low']:.4f}, {item['f1']['ci95_high']:.4f}] | {item['em']['point_effect']:.4f} | {item['judge']['point_effect']:.4f} |")
    effects = [bootstraps[str(b)]["metrics"]["f1"]["point_effect"] for b in BUDGETS]
    tighter_coherent = effects[0] >= effects[1] >= effects[2]
    (P25 / "MAIN1500_BUDGET_CURVE_RESULTS.md").write_text("# MAIN1500 Clean-Arm Budget Dependence\n\n" + "\n".join(curve) + f"\n\nPrespecified phrase ‘larger under tighter compression’ supported coherently: **{'YES' if tighter_coherent else 'NO'}**. The 256-token endpoint remains primary regardless.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "primary": primary, "tighter_compression_coherent": tighter_coherent}, indent=2))


if __name__ == "__main__":
    main()
