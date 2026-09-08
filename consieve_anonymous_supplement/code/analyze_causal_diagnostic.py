#!/usr/bin/env python3
"""Aggregate the frozen M3 stress runs and paired-bootstrap causal contrasts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from analyze_m2_stress import aggregate, percentile


BOOTSTRAP_SEED = 20260811
METRICS = [
    "positive_evidence_recall",
    "invalid_evidence_retention",
    "constraint_rejection_rate",
    "score_gap_mean",
    "pairwise_discrimination_accuracy",
    "auroc",
    "auprc",
    "constraint_sensitivity_delta",
    "positive_empty_rate",
    "perturbed_empty_rate",
]
CORE_VARIANTS = {"no_perturb", "verified_perturb"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_run_key(value: str) -> tuple[str, int]:
    variant, marker, seed = value.rpartition("_seed")
    if not marker or not variant or not seed.isdigit():
        raise ValueError(f"Expected <variant>_seed<int>, got {value!r}")
    return variant, int(seed)


def source_groups(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        ("all", rows),
        ("streamingqa", [row for row in rows if row["source"] == "streamingqa"]),
        ("timeqa", [row for row in rows if row["source"] == "timeqa"]),
    ]


def resampled_aggregate(
    by_id: dict[str, dict[str, Any]], sampled_ids: list[str]
) -> dict[str, Any]:
    return aggregate([by_id[pair_id] for pair_id in sampled_ids])


def paired_bootstrap_multi_seed(
    left_runs: list[dict[str, dict[str, Any]]],
    right_runs: list[dict[str, dict[str, Any]]],
    pair_ids: list[str],
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    if not pair_ids:
        return {
            metric: {
                "n_pairs": 0,
                "estimate": None,
                "ci95_low": None,
                "ci95_high": None,
            }
            for metric in METRICS
        }
    observed_left = [aggregate([run[pair_id] for pair_id in pair_ids]) for run in left_runs]
    observed_right = [aggregate([run[pair_id] for pair_id in pair_ids]) for run in right_runs]
    estimates = {
        metric: mean(float(item[metric]) for item in observed_left)
        - mean(float(item[metric]) for item in observed_right)
        for metric in METRICS
    }
    draws: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    n = len(pair_ids)
    for _ in range(iterations):
        sampled = [pair_ids[rng.randrange(n)] for _ in range(n)]
        left = [resampled_aggregate(run, sampled) for run in left_runs]
        right = [resampled_aggregate(run, sampled) for run in right_runs]
        for metric in METRICS:
            draws[metric].append(
                mean(float(item[metric]) for item in left)
                - mean(float(item[metric]) for item in right)
            )
    return {
        metric: {
            "n_pairs": n,
            "estimate": estimates[metric],
            "ci95_low": percentile(draws[metric], 0.025),
            "ci95_high": percentile(draws[metric], 0.975),
        }
        for metric in METRICS
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-dir", type=Path, default=root / "reports/m3_raw_stress_scores"
    )
    parser.add_argument(
        "--results", type=Path, default=root / "reports/m3_stress_ablation.csv"
    )
    parser.add_argument(
        "--seed-summary",
        type=Path,
        default=root / "reports/m3_stress_seed_summary.csv",
    )
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=root / "reports/m3_stress_paired_bootstrap.csv",
    )
    parser.add_argument(
        "--summary", type=Path, default=root / "reports/m3_stress_summary.json"
    )
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="Parallel comparison/source bootstrap jobs (sampling is unchanged).",
    )
    args = parser.parse_args()

    run_rows: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, tuple[str, int]] = {}
    for path in sorted(args.score_dir.glob("*.jsonl")):
        run_key = path.stem
        metadata[run_key] = parse_run_key(run_key)
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty score file: {path}")
        if any(row["model_key"] != run_key for row in rows):
            raise ValueError(f"model_key mismatch in {path}")
        if len({row["pair_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate pair_id in {path}")
        run_rows[run_key] = rows
    if not run_rows:
        raise ValueError(f"No score JSONL files under {args.score_dir}")

    reference_ids = {row["pair_id"] for row in next(iter(run_rows.values()))}
    for run_key, rows in run_rows.items():
        if {row["pair_id"] for row in rows} != reference_ids:
            raise ValueError(f"Frozen-pair membership mismatch for {run_key}")

    result_rows: list[dict[str, Any]] = []
    for run_key, rows in run_rows.items():
        variant, seed = metadata[run_key]
        for source, subset in source_groups(rows):
            result_rows.append(
                {
                    "run_key": run_key,
                    "variant": variant,
                    "seed": seed,
                    "source": source,
                    **aggregate(subset),
                }
            )
    write_csv(args.results, result_rows)

    seed_summary_rows: list[dict[str, Any]] = []
    for variant in sorted(CORE_VARIANTS):
        for source in ("all", "streamingqa", "timeqa"):
            subset = [
                row
                for row in result_rows
                if row["variant"] == variant and row["source"] == source
            ]
            if len(subset) < 2:
                continue
            output: dict[str, Any] = {
                "variant": variant,
                "source": source,
                "n_seeds": len(subset),
                "seeds": ",".join(str(row["seed"]) for row in sorted(subset, key=lambda x: x["seed"])),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in subset]
                output[f"{metric}_mean"] = mean(values)
                output[f"{metric}_std"] = stdev(values)
            seed_summary_rows.append(output)
    write_csv(args.seed_summary, seed_summary_rows)

    score_maps = {
        run_key: {row["pair_id"]: row for row in rows}
        for run_key, rows in run_rows.items()
    }
    comparisons = [
        (
            "verified_perturb_mean_minus_no_perturb_mean",
            [score_maps[f"verified_perturb_seed{seed}"] for seed in (42, 43, 44)],
            [score_maps[f"no_perturb_seed{seed}"] for seed in (42, 43, 44)],
        ),
        (
            "verified_perturb_seed42_minus_random_negative_seed42",
            [score_maps["verified_perturb_seed42"]],
            [score_maps["random_negative_seed42"]],
        ),
        (
            "verified_perturb_seed42_minus_retrieval_mined_seed42",
            [score_maps["verified_perturb_seed42"]],
            [score_maps["retrieval_mined_seed42"]],
        ),
        (
            "verified_perturb_seed42_minus_raw_perturb_seed42",
            [score_maps["verified_perturb_seed42"]],
            [score_maps["raw_perturb_seed42"]],
        ),
    ]
    bootstrap_rows: list[dict[str, Any]] = []
    reference_map = next(iter(score_maps.values()))
    bootstrap_jobs = []
    for comparison_index, (comparison, left, right) in enumerate(comparisons):
        for source_index, source in enumerate(("all", "streamingqa", "timeqa")):
            ids = sorted(
                pair_id
                for pair_id, row in reference_map.items()
                if source == "all" or row["source"] == source
            )
            bootstrap_jobs.append(
                (
                    comparison,
                    source,
                    left,
                    right,
                    ids,
                    BOOTSTRAP_SEED + comparison_index * 10 + source_index,
                )
            )
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                paired_bootstrap_multi_seed,
                left,
                right,
                ids,
                args.iterations,
                seed,
            )
            for _, _, left, right, ids, seed in bootstrap_jobs
        ]
        # Consume in declaration order so output row order remains deterministic.
        for (comparison, source, _, _, _, seed), future in zip(
            bootstrap_jobs, futures
        ):
            boot = future.result()
            for metric in METRICS:
                bootstrap_rows.append(
                    {
                        "comparison": comparison,
                        "source": source,
                        "metric": metric,
                        "iterations": args.iterations,
                        "seed": seed,
                        **boot[metric],
                    }
                )
    write_csv(args.bootstrap, bootstrap_rows)

    summary = {
        "schema_version": "m3_stress_summary_v1",
        "score_dir": str(args.score_dir),
        "runs": sorted(run_rows),
        "frozen_pair_count": len(reference_ids),
        "metrics": METRICS,
        "bootstrap_iterations": args.iterations,
        "bootstrap_seed_base": BOOTSTRAP_SEED,
        "result_csv": str(args.results),
        "seed_summary_csv": str(args.seed_summary),
        "bootstrap_csv": str(args.bootstrap),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
