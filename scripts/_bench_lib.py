"""Helpers shared by benchmark and plotting scripts."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml


LOGGER = logging.getLogger("bench")


_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class ModelSpec:
    """One side of the draft/target pair.

    Attributes:
        name: HuggingFace model id.
        device: Torch device string.
        dtype: Torch dtype.
    """

    name: str
    device: str
    dtype: torch.dtype


@dataclass(frozen=True)
class BenchConfig:
    """Parsed benchmark configuration.

    Attributes:
        draft: Draft model spec.
        target: Target model spec.
        K: Default speculative draft length.
        max_new_tokens: Default max tokens per generation.
        temperature: Default sampling temperature.
        num_prompts: Default number of prompts per dataset.
        cache_budget_B: Default Saguaro cache budget passed to SSDConfig.
        raw: Original YAML mapping for metadata dumping.
    """

    draft: ModelSpec
    target: ModelSpec
    K: int
    max_new_tokens: int
    temperature: float
    num_prompts: int
    cache_budget_B: int
    raw: dict[str, Any]


def _parse_dtype(value: str) -> torch.dtype:
    """Convert a YAML dtype string into a torch dtype.

    Args:
        value: Short dtype identifier (e.g. ``"bfloat16"``).

    Returns:
        The matching ``torch.dtype``.

    Raises:
        ValueError: When ``value`` is not a recognised dtype string.
    """
    key = value.lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype: {value!r}")
    return _DTYPE_MAP[key]


def load_config(path: Path) -> BenchConfig:
    """Read a YAML benchmark configuration from disk.

    Args:
        path: Path to the YAML config.

    Returns:
        A parsed :class:`BenchConfig` instance.
    """
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    draft_raw = raw["draft"]
    target_raw = raw["target"]
    defaults = raw.get("defaults", {})

    draft = ModelSpec(
        name=draft_raw["name"],
        device=draft_raw.get("device", "cuda:0"),
        dtype=_parse_dtype(draft_raw.get("dtype", "bfloat16")),
    )
    target = ModelSpec(
        name=target_raw["name"],
        device=target_raw.get("device", "cuda:1"),
        dtype=_parse_dtype(target_raw.get("dtype", "bfloat16")),
    )

    return BenchConfig(
        draft=draft,
        target=target,
        K=int(defaults.get("K", 4)),
        max_new_tokens=int(defaults.get("max_new_tokens", 256)),
        temperature=float(defaults.get("temperature", 1.0)),
        num_prompts=int(defaults.get("num_prompts", 50)),
        cache_budget_B=int(defaults.get("cache_budget_B", 8)),
        raw=raw,
    )


def make_run_dir(base: Path) -> Path:
    """Create and return ``base / run_<UTC-ISO>``.

    Args:
        base: Parent directory for the timestamped run directory.

    Returns:
        Path to the freshly created run directory.
    """
    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _git_sha() -> Optional[str]:
    """Return the current ``HEAD`` SHA or ``None`` when git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _nvidia_smi() -> Optional[str]:
    """Return ``nvidia-smi`` output or ``None`` when the tool is unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _versions() -> dict[str, str]:
    """Collect installed-package versions relevant to the run.

    Returns:
        Mapping of package name to version string.
    """
    versions: dict[str, str] = {"torch": torch.__version__}
    try:
        import transformers as _tf  # type: ignore

        versions["transformers"] = _tf.__version__
    except Exception:  # noqa: BLE001
        versions["transformers"] = "missing"
    try:
        import accelerate as _ac  # type: ignore

        versions["accelerate"] = _ac.__version__
    except Exception:  # noqa: BLE001
        versions["accelerate"] = "missing"
    return versions


def write_metadata(
    run_dir: Path,
    config: BenchConfig,
    cli_args: dict[str, Any],
) -> Path:
    """Write ``metadata.json`` describing the run environment.

    Args:
        run_dir: Directory in which to write the file.
        config: Parsed benchmark configuration.
        cli_args: Effective CLI argument mapping.

    Returns:
        Path to the written metadata file.
    """
    meta = {
        "git_sha": _git_sha(),
        "nvidia_smi": _nvidia_smi(),
        "versions": _versions(),
        "config": config.raw,
        "cli_args": cli_args,
    }
    path = run_dir / "metadata.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return path


def load_prompts(data_dir: Path, dataset: str, limit: int) -> list[dict[str, Any]]:
    """Load up to ``limit`` prompt rows for ``dataset`` from JSONL on disk.

    Args:
        data_dir: Directory containing ``prompts_<dataset>.jsonl`` files.
        dataset: Dataset short name.
        limit: Maximum number of prompts to return.

    Returns:
        List of prompt dictionaries (possibly empty).
    """
    path = data_dir / f"prompts_{dataset}.jsonl"
    if not path.exists():
        LOGGER.warning("missing prompts file: %s", path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows
