"""Clean Evaluation v2 invariants and deterministic manifest selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_VERSION = "clean_v2"
LEGACY_PROTOCOL_VERSION = "legacy_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def select_entries_by_manifest(entries: list[Any], manifest_path: str | Path) -> list[Any]:
    """Select entries in manifest order and verify ID/index/hash identity."""
    manifest = load_json(manifest_path)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Manifest is not {PROTOCOL_VERSION}: {manifest_path}")

    selected = []
    for sample in manifest.get("samples", []):
        index = sample["index"]
        if index < 0 or index >= len(entries):
            raise IndexError(f"Manifest index out of range: {index}")
        entry = entries[index]
        if str(getattr(entry, "id", "")) != str(sample["id"]):
            raise ValueError(f"Manifest ID mismatch at index {index}")
        selected.append(entry)
    return selected


def assert_clean_v2_config(config: dict[str, Any]) -> None:
    """Fail closed if a clean run could read a legacy cache or first-N sample."""
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Clean evaluation requires protocol_version=clean_v2")
    dataset = config.get("dataset", {})
    if dataset.get("eval_samples") not in (None, 0):
        raise ValueError("clean_v2 forbids first-N eval_samples; use a manifest")
    if not dataset.get("manifest_path"):
        raise ValueError("clean_v2 requires dataset.manifest_path")

    cache_paths = [
        config.get("llm", {}).get("cache_path"),
        config.get("judge", {}).get("cache_path"),
        config.get("retrieval", {}).get("bm25_cache_path"),
        config.get("retrieval", {}).get("reranker", {}).get("cache_path"),
        config.get("highlighter", {}).get("cache_path"),
    ]
    for raw_path in filter(None, cache_paths):
        normalized = str(raw_path).replace("\\", "/")
        if "/clean_v2/" not in f"/{normalized.lstrip('./')}":
            raise ValueError(f"clean_v2 cache path is outside clean_v2 namespace: {raw_path}")

    highlighter_cfg = config.get("highlighter", {})
    configured_models = highlighter_cfg.get("models", [])
    thresholds = highlighter_cfg.get("thresholds", {})
    model_aliases = highlighter_cfg.get("model_aliases", {})
    for model_id in configured_models:
        if thresholds.get(model_id, highlighter_cfg.get("threshold")) is None:
            raise ValueError(
                f"clean_v2 requires a validation-frozen threshold for model: {model_id}"
            )
    aliases = [model_aliases.get(model_id) for model_id in configured_models]
    explicit_aliases = [alias for alias in aliases if alias is not None]
    if any(not str(alias).strip() for alias in explicit_aliases):
        raise ValueError("clean_v2 highlighter model aliases must be non-empty")
    if len(explicit_aliases) != len(set(explicit_aliases)):
        raise ValueError("clean_v2 highlighter model aliases must be unique")


def apply_manifest_if_configured(
    entries: list[Any], dataset_config: dict[str, Any]
) -> list[Any]:
    manifest_path = dataset_config.get("manifest_path")
    if not manifest_path:
        return entries
    return select_entries_by_manifest(entries, manifest_path)
