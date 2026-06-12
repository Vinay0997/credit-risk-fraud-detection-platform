"""Configuration loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PlatformConfig:
    raw: dict[str, Any]

    @property
    def random_seed(self) -> int:
        return int(self.raw.get("models", {}).get("random_seed", 42))

    @property
    def credit_target(self) -> str:
        return str(self.raw.get("models", {}).get("credit_target", "default_label"))

    @property
    def fraud_target(self) -> str:
        return str(self.raw.get("models", {}).get("fraud_target", "fraud_label"))


def load_config(path: str | Path = "config/app_config.yaml") -> PlatformConfig:
    config_path = Path(path)
    if not config_path.exists():
        return PlatformConfig(raw={})
    with config_path.open("r", encoding="utf-8") as handle:
        return PlatformConfig(raw=yaml.safe_load(handle) or {})
