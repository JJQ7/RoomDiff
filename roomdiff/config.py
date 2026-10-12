"""Configuration: `configs/default.yaml`, optionally overridden by a second YAML file
and by `section.key=value` strings from the command line.

The default file doubles as the schema: overrides may only set keys that already exist,
with a value of the same type (an int is accepted for a float; a null default accepts
anything; an "auto" default accepts a number), so a typo fails loudly instead of
being silently ignored.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Unknown key, wrong value type or malformed override."""


class Config(dict):
    """Nested dict with attribute access: `cfg.ingest.max_side`."""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = Config(value) if isinstance(value, Mapping) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"config has no key {name!r}") from None

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict (for YAML/JSON dumps)."""
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}


def load_config(
    path: str | Path | None = None,
    override_file: str | Path | None = None,
    overrides: Iterable[str] = (),
) -> Config:
    """Load `path` (default: configs/default.yaml), apply `override_file`, then apply
    `overrides` such as `"ingest.num_keyframes=32"` in order (later ones win)."""
    cfg = _read_yaml(path or DEFAULT_CONFIG)
    if override_file is not None:
        _merge(cfg, _read_yaml(override_file))
    for item in overrides:
        dotted, value = parse_override(item)
        for part in reversed(dotted.split(".")):  # "a.b" -> {"a": {"b": value}}
            value = {part: value}
        _merge(cfg, value)
    return Config(cfg)


def parse_override(item: str) -> tuple[str, Any]:
    """Split `"ingest.num_keyframes=32"` into `("ingest.num_keyframes", 32)`."""
    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise ConfigError(f"override must look like 'section.key=value', got {item!r}")
    return key.strip(), coerce(raw)


def coerce(text: str) -> bool | int | float | str | None:
    """Type a command-line value: true/false -> bool, null/none -> None, then int, then
    float (`1e-3` included, which YAML 1.1 would leave a string), else the string."""
    text = text.strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level of a config file must be a mapping")
    return data


def _merge(base: dict[str, Any], update: Mapping[str, Any], prefix: str = "") -> None:
    """Recursively write `update` into `base` (existing keys and types only)."""
    for key, value in update.items():
        dotted = f"{prefix}{key}"
        if key not in base:
            raise ConfigError(f"unknown config key {dotted!r}")
        old = base[key]
        if isinstance(old, dict):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{dotted!r} is a section; set one of its keys instead")
            _merge(old, value, prefix=f"{dotted}.")
            continue
        if isinstance(old, float) and type(value) is int:
            value = float(value)
        if old is not None and type(value) is not type(old) and not _auto_or_number(old, value):
            raise ConfigError(f"{dotted!r} must be {type(old).__name__}, got {value!r}")
        base[key] = value


def _auto_or_number(old: Any, value: Any) -> bool:
    """A key whose default is "auto" also accepts a number."""
    return old == "auto" and isinstance(value, int | float) and not isinstance(value, bool)
