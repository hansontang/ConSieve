"""Append-only audit logging for external LLM API calls.

The logger intentionally stores prompts and responses for reproducibility, but
redacts credentials and authentication headers before anything reaches disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_SENSITIVE_KEYS = re.compile(
    r"(?:api[_-]?key|authorization|access[_-]?token|secret|password|cookie)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_DASHSCOPE_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def redact(value: Any, key: str = "") -> Any:
    """Recursively remove secrets while retaining reproducibility metadata."""
    if _SENSITIVE_KEYS.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _DASHSCOPE_KEY.sub("[REDACTED]", _BEARER.sub("Bearer [REDACTED]", value))
    return value


def response_metadata(response: Any) -> Dict[str, Any]:
    """Extract stable OpenAI-compatible response metadata without credentials."""
    usage = getattr(response, "usage", None)
    usage_dict: Dict[str, Any] = {}
    if usage is not None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_dict[name] = int(getattr(usage, name, 0) or 0)
        details = getattr(usage, "model_dump", None)
        if callable(details):
            usage_dict["provider_details"] = redact(details())

    choices = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        choices.append(
            {
                "index": getattr(choice, "index", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "content": getattr(message, "content", None),
                "refusal": getattr(message, "refusal", None),
            }
        )
    return redact(
        {
            "provider_request_id": getattr(response, "id", None),
            "created": getattr(response, "created", None),
            "model": getattr(response, "model", None),
            "object": getattr(response, "object", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "usage": usage_dict,
            "choices": choices,
        }
    )


class APIAuditLogger:
    """Thread-safe, append-only JSONL logger with one metadata file per run."""

    def __init__(
        self,
        log_dir: str,
        run_id: str,
        stage: str,
        provider: str = "qwen_dashscope",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_id = run_id
        self.stage = stage
        self.provider = provider
        self.run_dir = Path(log_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.requests_path = self.run_dir / "requests.jsonl"
        self.metadata_path = self.run_dir / "run_metadata.json"
        self._lock = threading.Lock()
        self._counter = 0
        self._write_run_metadata(extra_metadata or {})

    def _write_run_metadata(self, extra: Dict[str, Any]) -> None:
        if self.metadata_path.exists():
            return
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except Exception:
            commit = ""
        metadata = redact(
            {
                "schema_version": "ds_api_audit_v1",
                "run_id": self.run_id,
                "stage": self.stage,
                "provider": self.provider,
                "created_at_utc": utc_now(),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "python": sys.version,
                "argv": sys.argv,
                "git_commit": commit or None,
                "log_file": str(self.requests_path),
                "stores_full_requests": True,
                "stores_full_responses": True,
                "credentials_redacted": True,
                "extra": extra,
            }
        )
        self.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def next_request_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"{self.run_id}-{self._counter:06d}-{uuid.uuid4().hex[:8]}"

    def log(
        self,
        *,
        request_id: str,
        request: Dict[str, Any],
        started_at: float,
        status: str,
        response: Optional[Any] = None,
        error: Optional[str] = None,
        attempt: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_request = redact(request)
        canonical_request = json.dumps(safe_request, ensure_ascii=False, sort_keys=True)
        event = {
            "schema_version": "ds_api_audit_v1",
            "timestamp_utc": utc_now(),
            "run_id": self.run_id,
            "stage": self.stage,
            "provider": self.provider,
            "request_id": request_id,
            "attempt": attempt,
            "status": status,
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "request_sha256": hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            "request": safe_request,
            "response": response_metadata(response) if response is not None else None,
            "error": redact(error) if error else None,
            "metadata": redact(metadata or {}),
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with self.requests_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()


def audit_logger_from_config(config: Dict[str, Any]) -> Optional[APIAuditLogger]:
    """Build a logger from an ``api_audit`` config block or environment."""
    cfg = config.get("api_audit", {}) or {}
    enabled = cfg.get("enabled", bool(os.environ.get("DS_API_AUDIT_DIR")))
    if not enabled:
        return None
    log_dir = cfg.get("log_dir") or os.environ.get("DS_API_AUDIT_DIR")
    if not log_dir:
        raise ValueError("API audit is enabled but no log_dir/DS_API_AUDIT_DIR is set")
    run_id = cfg.get("run_id") or os.environ.get("DS_API_RUN_ID")
    if not run_id:
        raise ValueError("API audit is enabled but no run_id/DS_API_RUN_ID is set")
    return APIAuditLogger(
        log_dir=log_dir,
        run_id=run_id,
        stage=cfg.get("stage") or os.environ.get("DS_API_STAGE", "unspecified"),
        provider=cfg.get("provider", "qwen_dashscope"),
        extra_metadata=cfg.get("metadata", {}),
    )
