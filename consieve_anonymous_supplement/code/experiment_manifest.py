"""Immutable experiment-manifest writer for clean_v2 outputs."""

from __future__ import annotations

import datetime as dt
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .clean_protocol import PROTOCOL_VERSION, sha256_file
from .config_utils import redact_secrets
from .query_formatter import QUERY_FORMATTER_VERSION


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_experiment_manifest(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    manifest_path = dataset_cfg.get("manifest_path")
    dataset_path = dataset_cfg.get("qa_path")
    retrieval_cfg = config.get("retrieval", {})
    index_path = Path(retrieval_cfg.get("index_path", ""))
    index_manifest = index_path / "clean_v2_index_manifest.json"
    index_metadata: dict[str, Any] = {}
    if index_manifest.exists():
        try:
            index_metadata = json.loads(index_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index_metadata = {}
    token_cfg = config.get("token_accounting", {})
    highlighter_cfg = config.get("highlighter", {})

    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "command": shlex.join(sys.argv),
        "config": redact_secrets(config),
        "dataset": {
            "path": dataset_path,
            "sha256": sha256_file(dataset_path) if dataset_path and Path(dataset_path).exists() else None,
        },
        "evaluation_manifest": {
            "path": manifest_path,
            "sha256": sha256_file(manifest_path) if manifest_path and Path(manifest_path).exists() else None,
        },
        "models": highlighter_cfg.get("models", []),
        "model_revisions": highlighter_cfg.get("model_revisions", {}),
        "tokenizer": {
            "name": token_cfg.get("tokenizer_name"),
            "revision": token_cfg.get("tokenizer_revision"),
        },
        "query_formatter_version": QUERY_FORMATTER_VERSION,
        "retrieval_index": {
            "path": str(index_path) if str(index_path) else None,
            "manifest_path": str(index_manifest) if index_manifest.exists() else None,
            "manifest_sha256": sha256_file(index_manifest) if index_manifest.exists() else None,
            "corpus_checksum": index_metadata.get(
                "corpus_checksum", retrieval_cfg.get("corpus_checksum")
            ),
        },
        "threshold": highlighter_cfg.get("threshold"),
        "thresholds": highlighter_cfg.get("thresholds", {}),
        "threshold_selection": highlighter_cfg.get("threshold_selection", {}),
        "generator": redact_secrets(config.get("llm", {})),
        "judge": redact_secrets(config.get("judge", {})),
        "random_seed": config.get("random_seed"),
        "expected_outputs": ["raw per-example JSON", "aggregate metrics JSON"],
    }


def write_experiment_manifest(config: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment_manifest.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(build_experiment_manifest(config), handle, ensure_ascii=False, indent=2)
    return path
