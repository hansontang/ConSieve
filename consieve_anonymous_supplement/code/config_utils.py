"""Configuration helpers that never persist or print secret values."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any


_ENV_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$")
_SECRET_KEYS = ("api_key", "token", "secret", "password")


def resolve_environment(value: Any, *, strict: bool = True) -> Any:
    """Recursively resolve exact `${VAR}` and `${VAR:-default}` values."""
    if isinstance(value, dict):
        return {key: resolve_environment(item, strict=strict) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_environment(item, strict=strict) for item in value]
    if not isinstance(value, str):
        return value

    match = _ENV_RE.match(value)
    if not match:
        return value
    name, default = match.groups()
    if name in os.environ:
        return os.environ[name]
    if default is not None:
        return default
    if strict:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def redact_secrets(value: Any) -> Any:
    """Return a deep redacted copy suitable for experiment manifests."""
    value = deepcopy(value)

    def walk(item: Any) -> Any:
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                if any(marker in key.lower() for marker in _SECRET_KEYS):
                    result[key] = "<redacted>"
                else:
                    result[key] = walk(child)
            return result
        if isinstance(item, list):
            return [walk(child) for child in item]
        return item

    return walk(value)

