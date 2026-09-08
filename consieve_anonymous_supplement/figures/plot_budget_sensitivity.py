#!/usr/bin/env python3
"""Render the Phase 10.14 four-curve GT406 budget-sensitivity figure.

This script performs no model inference and no metric estimation. It copies
registered aggregate rows from frozen artifacts into one plotting table.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT = Path("${CONSIEVE_ROOT}")
OUT = ROOT / "paper_rewrite/phase10_14/figures"
M4 = ROOT / "reports/m4_matched_budget.csv"
SENTINEL = ROOT / "paper_rewrite/phase9/reports/sentinel_gt406_curve.csv"

EXPECTED = {
    M4: "4fdef18316f53b3e1f4cf9eb771b899b3bd80abb2a17143cdc8670df59ba4ed0",
    SENTINEL: "f91f5f66f12d496a68c1c024939b7f559ccb6a2af4fb9530d6d39562daa1240a",
}
BUDGETS = (64, 128, 256, 512, 1024)
CURVES = ("OpenProvence", "ConSieve", "EXIT", "Sentinel")
STYLE = {
    "OpenProvence": {
        "color": "#009E73", "marker": "s", "linewidth": 1.65,
        "linestyle": (0, (4.0, 1.6)),
    },
    "ConSieve": {
        "color": "#0072B2", "marker": "o", "linewidth": 2.25,
        "linestyle": "-",
    },
    "EXIT": {
        "color": "#E69F00", "marker": "^", "linewidth": 1.65,
        "linestyle": (0, (5.0, 1.5, 1.1, 1.5)),
    },
    "Sentinel": {
        "color": "#D55E00", "marker": "D", "linewidth": 1.65,
        "linestyle": (0, (1.2, 1.4)),
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def checked_sources() -> None:
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen source hash mismatch: {path}: {actual}")


def build_points() -> list[dict[str, object]]:
    checked_sources()
    m4 = rows(M4)
    sentinel = rows(SENTINEL)
    points: list[dict[str, object]] = []

    aliases = {
        "OpenProvence": "OpenProvence",
        "DS-ModernBERT": "ConSieve",
        "EXIT": "EXIT",
    }
    for source_name, display_name in aliases.items():
        selected = [
            row for row in m4
            if row["method"] == source_name
            and row["scenario"] == "BM25@25"
            and row["fairness_mode"] == "per_query_hard_cap"
            and int(float(row["budget"])) in BUDGETS
        ]
        if len(selected) != 5:
            raise ValueError(
                f"Expected five M4 rows for {source_name}, got {len(selected)}"
            )
        for row in selected:
            points.append({
                "method": display_name,
                "budget": int(float(row["budget"])),
                "context_tokens": float(row["context_tokens"]),
                "f1": float(row["f1"]),
                "source_artifact": str(M4.relative_to(ROOT)),
                "source_sha256": EXPECTED[M4],
            })

    selected_sentinel = [
        row for row in sentinel
        if row["method"] == "Sentinel"
        and row["source"] == "all"
        and int(float(row["budget"])) in BUDGETS
    ]
    if len(selected_sentinel) != 5:
        raise ValueError("Frozen Phase-9 Sentinel curve is incomplete")
    for row in selected_sentinel:
        if int(float(row["n"])) != 406:
            raise ValueError("Sentinel denominator mismatch")
        points.append({
            "method": "Sentinel",
            "budget": int(float(row["budget"])),
            "context_tokens": float(row["mean_context_tokens"]),
            "f1": float(row["f1"]),
            "source_artifact": str(SENTINEL.relative_to(ROOT)),
            "source_sha256": EXPECTED[SENTINEL],
        })

    counts = {
        method: sum(point["method"] == method for point in points)
        for method in CURVES
    }
    if counts != {method: 5 for method in CURVES}:
        raise ValueError(f"Curve roster mismatch: {counts}")
    if any(point["method"] == "LongLLMLingua" for point in points):
        raise ValueError("LongLLMLingua must not enter the four-curve figure")
    return sorted(points, key=lambda p: (CURVES.index(p["method"]), p["budget"]))


def write_data(points: list[dict[str, object]]) -> Path:
    path = OUT / "figure2_gt406_fourcurve_data.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    return path


def render(points: list[dict[str, object]]) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8.7,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.9,
        "axes.linewidth": 0.72,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 400,
    })
    fig, ax = plt.subplots(figsize=(5.55, 2.62))

    for method in CURVES:
        part = sorted(
            (point for point in points if point["method"] == method),
            key=lambda point: point["budget"],
        )
        style = STYLE[method]
        ax.plot(
            [point["context_tokens"] for point in part],
            [point["f1"] for point in part],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=4.9 if method == "ConSieve" else 4.5,
            markeredgewidth=0.75,
            markeredgecolor="white",
            label=method,
            zorder=5 if method == "ConSieve" else 3,
        )

    ax.set_xscale("log")
    ax.set_xlim(43, 1100)
    ax.set_ylim(0.24, 0.675)
    ax.set_xticks([50, 100, 200, 500, 1000])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks(np.arange(0.25, 0.70, 0.05))
    ax.set_xlabel("Actual mean Qwen context tokens (log scale)")
    ax.set_ylabel("Downstream F1")
    ax.minorticks_off()
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.48, alpha=0.72)
    ax.grid(axis="x", which="major", color="#ECECEC", linewidth=0.42, alpha=0.68)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.65)

    ax.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        columnspacing=1.35,
        handlelength=2.15,
        handletextpad=0.48,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.215, top=0.82)

    stem = OUT / "figure2_gt406_fourcurve"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.035, dpi=400)
    plt.close(fig)


def write_provenance(points: list[dict[str, object]], data_path: Path) -> None:
    sentinel_dependencies = [
        ROOT / "paper_rewrite/phase9/results/sentinel/gt406_scores.manifest.json",
        ROOT / "paper_rewrite/phase9/results/sentinel/contexts/gt406/sentinel_contexts.manifest.json",
        *[
            ROOT / f"paper_rewrite/phase9/results/sentinel/qa/gt406_budget{budget}.json"
            for budget in BUDGETS
        ],
    ]
    payload = {
        "schema_version": "figure2_gt406_fourcurve_v1",
        "paper_result": True,
        "formal_experiment_run": False,
        "dataset": "GT406",
        "retrieval": "BM25@25",
        "roster_rule": "complete_64_128_256_512_1024_sweeps_only",
        "curves": {method: list(BUDGETS) for method in CURVES},
        "excluded_from_figure": {
            "LongLLMLingua": (
                "No complete GT406 five-budget sweep; retained in Table 1 and "
                "Appendix natural-point results."
            ),
        },
        "point_count": len(points),
        "frontier_drawn": False,
        "vertical_budget_line_drawn": False,
        "data_csv": str(data_path),
        "data_csv_sha256": sha256(data_path),
        "source_artifacts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in EXPECTED
        },
        "sentinel_dependencies": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sentinel_dependencies
        },
    }
    (OUT / "figure2_gt406_fourcurve.provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    points = build_points()
    data_path = write_data(points)
    render(points)
    write_provenance(points, data_path)
    print(json.dumps({
        "points": len(points),
        "curves": {method: 5 for method in CURVES},
        "longllmlingua_points": 0,
        "frontier_drawn": False,
        "pdf_sha256": sha256(OUT / "figure2_gt406_fourcurve.pdf"),
        "svg_sha256": sha256(OUT / "figure2_gt406_fourcurve.svg"),
    }, indent=2))


if __name__ == "__main__":
    main()
