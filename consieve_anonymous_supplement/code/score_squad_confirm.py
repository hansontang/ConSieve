#!/usr/bin/env python3
"""Fail-closed scoring for the frozen Phase-10.21 SQuAD confirmatory split.

This script emits full-precision evidence-sentence scores only.  It does not
select calibration thresholds or aggregate TEST results; those operations are
performed only after every requested score artifact is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("${CONSIEVE_ROOT}")
P21 = ROOT / "paper_rewrite/phase10_21"
P25 = ROOT / "paper_rewrite/phase10_25"
SHARED = ROOT / "paper_rewrite/phase10_17/scripts/score_garage_formal.py"
SHARED_SHA = "3cd5a7fb8e4eee46ce81cf9e7506419cd2e4febb87431efcb4b847eb2b16e7bd"
SPLIT_INFO = {
    "calib": {
        "path": P21 / "SQUAD_CONFIRM_CALIB_V1.jsonl",
        "sha256": "106404c78f0ffe970a431a4e9a964c5eb484d345d08a189a3be9c5d3b7ec26da",
        "rows": 1411,
        "primary": Counter({"ENTITY_SWAP": 607, "NEGATION": 259, "ANTONYM": 440, "NUMBER_SWAP_NON_TEMPORAL": 71}),
        "secondary": Counter({"MUTUAL_EXCLUSION": 34}),
        "contexts": 1035,
    },
    "test": {
        "path": P21 / "SQUAD_CONFIRM_TEST_V1.jsonl",
        "sha256": "696a6bf8b3d0b673163d2a4e5b92d448a9f46ed987faea8748014098ba68c89d",
        "rows": 3405,
        "primary": Counter({"ENTITY_SWAP": 1454, "NEGATION": 669, "ANTONYM": 1028, "NUMBER_SWAP_NON_TEMPORAL": 164}),
        "secondary": Counter({"MUTUAL_EXCLUSION": 90}),
        "contexts": 2414,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_shared():
    actual = sha256(SHARED)
    if actual != SHARED_SHA:
        raise RuntimeError(f"frozen shared scorer drift: {actual}")
    spec = importlib.util.spec_from_file_location("p1025_frozen_loaders", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen model loaders")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_rows(split: str, splitter) -> list[dict[str, Any]]:
    info = SPLIT_INFO[split]
    path = info["path"]
    actual = sha256(path)
    if actual != info["sha256"]:
        raise RuntimeError(f"sealed {split} SHA drift: {actual}")
    rows = read_jsonl(path)
    if len(rows) != info["rows"]:
        raise RuntimeError(f"sealed {split} row drift: {len(rows)}")
    primary, secondary, ids, contexts = Counter(), Counter(), set(), set()
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id in ids:
            raise RuntimeError(f"duplicate pair_id: {pair_id}")
        ids.add(pair_id)
        contexts.add(str(row["context_id"]))
        if row.get("model_scores") is not None or row.get("generated_gold_labels") is not None:
            raise RuntimeError(f"sealed row contains model/generated labels: {pair_id}")
        target = primary if row["analysis_role"] == "PRIMARY" else secondary
        target[row["cause_family"]] += 1
        spans = splitter.split(row["context"])
        index = int(row["evidence_sentence_index"])
        if not 0 <= index < len(spans):
            raise RuntimeError(f"evidence index out of range: {pair_id}")
        span = spans[index]
        if (span.text != row["evidence_sentence_text"] or span.start != int(row["evidence_sentence_start"]) or span.end != int(row["evidence_sentence_end"])):
            raise RuntimeError(f"frozen sentence identity drift: {pair_id}")
    if primary != info["primary"] or secondary != info["secondary"]:
        raise RuntimeError(f"family count drift: primary={primary}, secondary={secondary}")
    if len(contexts) != info["contexts"]:
        raise RuntimeError(f"context count drift: {len(contexts)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=sorted(SPLIT_INFO))
    parser.add_argument("--method-key", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--failure-log", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    freeze = P25 / "FORMAL_METHOD_FREEZE_MANIFEST.json"
    if not freeze.is_file():
        raise RuntimeError("formal method-freeze manifest does not exist")
    freeze_payload = json.loads(freeze.read_text(encoding="utf-8"))
    if freeze_payload.get("status") != "FROZEN" or not freeze_payload.get("all_six_validation_thresholds_frozen"):
        raise RuntimeError("formal method freeze is incomplete")
    if args.split == "test":
        calibration_registry = P25 / "SQUAD_CONFIRM_CALIBRATION_REGISTRY.json"
        if not calibration_registry.is_file():
            raise RuntimeError("TEST scoring is sealed until the calibration registry is frozen")
        calibration_payload = json.loads(calibration_registry.read_text(encoding="utf-8"))
        if calibration_payload.get("status") != "FROZEN" or args.method_key not in calibration_payload.get("methods", {}):
            raise RuntimeError("TEST scoring requested before this method's calibration threshold was frozen")
    if args.output.exists() or Path(str(args.output) + ".partial").exists():
        raise RuntimeError("refusing to overwrite score output")
    if args.failure_log.exists() and args.failure_log.stat().st_size:
        raise RuntimeError("non-empty failure log exists")

    shared = load_shared()
    splitter = shared.load_splitter()
    rows = validate_rows(args.split, splitter)
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        raise RuntimeError(f"missing model path: {model_path}")
    model_hash = sha256(model_path / "model.safetensors")
    if args.method_key in freeze_payload.get("checkpoints", {}):
        expected_model_hash = freeze_payload["checkpoints"][args.method_key]["final_model_sha256"]
    elif args.method_key == "consieve":
        expected_model_hash = freeze_payload["historical_methods"]["canonical_consieve"]["model_sha256"]
    elif args.method_key == "openprovence":
        expected_model_hash = freeze_payload["historical_methods"]["openprovence"]["model_sha256"]
    else:
        raise RuntimeError(f"method is absent from the formal freeze: {args.method_key}")
    if model_hash != expected_model_hash:
        raise RuntimeError(f"frozen model hash mismatch for {args.method_key}: {model_hash}")
    loader_revision = args.model_revision or model_path.name
    config = {
        "display_name": args.display_name,
        "kind": "highlighter",
        "model_id": str(model_path),
        "revision": loader_revision,
        "snapshot": str(model_path),
        "parameter_count": 315_204_867,
    }
    model, scorer = shared.load_highlighter(config, args.device, args.batch_size)
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failure_log.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(args.output) + ".partial")
    latencies: list[float] = []
    started_all = time.perf_counter()
    with partial.open("x", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows, start=1):
            pair_id = str(row["pair_id"])
            try:
                spans = splitter.split(row["context"])
                sentences = [span.text for span in spans]
                evidence_index = int(row["evidence_sentence_index"])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                qplus_scores, qplus_audit = scorer(row["qplus_question"], row["context"], sentences)
                qminus_scores, qminus_audit = scorer(row["qminus_question"], row["context"], sentences)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                qplus = float(qplus_scores[evidence_index])
                qminus = float(qminus_scores[evidence_index])
                if not math.isfinite(qplus) or not math.isfinite(qminus):
                    raise RuntimeError("non-finite score")
                if qplus_audit.get("truncation_applied") or qminus_audit.get("truncation_applied"):
                    raise RuntimeError("unauthorized truncation")
                payload = {
                    "schema_version": "phase10_25_squad_confirm_scores/1.0",
                    "confirm_split": args.split.upper(),
                    "pair_id": pair_id,
                    "context_id": str(row["context_id"]),
                    "analysis_role": row["analysis_role"],
                    "cause_code": row["cause_code"],
                    "cause_family": row["cause_family"],
                    "method_key": args.method_key,
                    "display_name": args.display_name,
                    "model_path": str(model_path),
                    "model_revision": args.model_revision,
                    "evidence_sentence_index": evidence_index,
                    "qplus_score_full_precision": qplus,
                    "qminus_score_full_precision": qminus,
                    "qplus_score_hex": qplus.hex(),
                    "qminus_score_hex": qminus.hex(),
                    "runtime_seconds": elapsed,
                    "qplus_scoring_audit": qplus_audit,
                    "qminus_scoring_audit": qminus_audit,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                latencies.append(elapsed)
                if row_index == 1 or row_index % 100 == 0 or row_index == len(rows):
                    print(f"{args.method_key} {args.split}: {row_index}/{len(rows)}", flush=True)
            except Exception as error:
                failure = {
                    "pair_id": pair_id,
                    "method_key": args.method_key,
                    "split": args.split,
                    "error_class": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
                with args.failure_log.open("a", encoding="utf-8") as failure_handle:
                    failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                raise
    partial.replace(args.output)
    ordered = sorted(latencies)
    manifest = {
        "schema_version": "phase10_25_squad_score_manifest/1.0",
        "method_key": args.method_key,
        "display_name": args.display_name,
        "split": args.split,
        "input": str(SPLIT_INFO[args.split]["path"]),
        "input_sha256": SPLIT_INFO[args.split]["sha256"],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(rows),
        "contexts": len({str(row["context_id"]) for row in rows}),
        "model_path": str(model_path),
        "model_safetensors_sha256": model_hash,
        "model_revision": args.model_revision,
        "elapsed_seconds": time.perf_counter() - started_all,
        "latency_mean_seconds_per_pair": sum(latencies) / len(latencies),
        "latency_median_seconds_per_pair": ordered[len(ordered) // 2],
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "unresolved_failures": 0,
        "formal_method_freeze_manifest_sha256": sha256(freeze),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
