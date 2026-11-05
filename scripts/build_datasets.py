"""Materialise benchmark prompts from public HuggingFace datasets as JSONL.

For each supported dataset, samples ``--num-prompts`` items deterministically
with the given ``--seed`` and writes ``data/prompts_<name>.jsonl`` with rows
of the form ``{"id": int, "dataset": str, "prompt": str}``.

The JSONL files are intended to be committed so the benchmark host does not
need access to the HuggingFace datasets API at run time.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from datasets import load_dataset


LOGGER = logging.getLogger("datasets-builder")


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for one benchmark dataset.

    Attributes:
        name: Short dataset key used in output file names and ``dataset`` field.
        hf_path: HuggingFace dataset path.
        hf_config: Optional dataset configuration name.
        split: Dataset split to load.
        prompt_field_candidates: Ordered list of fields to try as the prompt.
        sample: If True, take a random ``num_prompts`` sample; if False,
            slice the first ``num_prompts`` rows.
    """

    name: str
    hf_path: str
    hf_config: Optional[str]
    split: str
    prompt_field_candidates: tuple[str, ...]
    sample: bool


SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="humaneval",
        hf_path="openai_humaneval",
        hf_config=None,
        split="test",
        prompt_field_candidates=("prompt",),
        sample=False,
    ),
    DatasetSpec(
        name="alpaca",
        hf_path="tatsu-lab/alpaca",
        hf_config=None,
        split="train",
        prompt_field_candidates=("instruction",),
        sample=True,
    ),
    DatasetSpec(
        name="ultrafeedback",
        hf_path="openbmb/UltraFeedback",
        hf_config=None,
        split="train",
        prompt_field_candidates=("instruction", "prompt"),
        sample=True,
    ),
    DatasetSpec(
        name="gsm8k",
        hf_path="gsm8k",
        hf_config="main",
        split="train",
        prompt_field_candidates=("question",),
        sample=True,
    ),
)


def _pick_field(row: dict, candidates: Iterable[str]) -> Optional[str]:
    """Return the first present, non-empty string field from ``candidates``.

    Args:
        row: A dataset row exposed as a mapping.
        candidates: Ordered field names to probe.

    Returns:
        The matching string value, or ``None`` if no candidate exists.
    """
    for field in candidates:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def build_dataset(
    spec: DatasetSpec,
    out_dir: Path,
    num_prompts: int,
    seed: int,
    refresh: bool,
) -> Optional[Path]:
    """Materialise one dataset into a JSONL file under ``out_dir``.

    Args:
        spec: Dataset specification.
        out_dir: Directory in which to write ``prompts_<name>.jsonl``.
        num_prompts: Number of prompts to keep.
        seed: Deterministic sampling seed.
        refresh: When True, overwrite an existing file; when False, skip.

    Returns:
        The output path on success, or ``None`` when the file was skipped or
        the dataset failed to load.
    """
    out_path = out_dir / f"prompts_{spec.name}.jsonl"
    if out_path.exists() and not refresh:
        LOGGER.info("skip %s (exists)", out_path)
        return None

    LOGGER.info("loading %s split=%s config=%s", spec.hf_path, spec.split, spec.hf_config)
    try:
        if spec.hf_config is not None:
            ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
        else:
            ds = load_dataset(spec.hf_path, split=spec.split)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("failed to load %s: %s", spec.hf_path, exc)
        return None

    indices: list[int]
    n = len(ds)
    if spec.sample:
        rng = random.Random(seed)
        indices = rng.sample(range(n), k=min(num_prompts, n))
    else:
        indices = list(range(min(num_prompts, n)))

    rows: list[dict] = []
    for new_id, idx in enumerate(indices):
        row = ds[idx]
        prompt = _pick_field(row, spec.prompt_field_candidates)
        if prompt is None:
            continue
        rows.append({"id": new_id, "dataset": spec.name, "prompt": prompt})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    LOGGER.info("wrote %d prompts to %s", len(rows), out_path)
    return out_path


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the dataset builder.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the dataset builder for all configured specs.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        build_dataset(
            spec=spec,
            out_dir=args.out,
            num_prompts=args.num_prompts,
            seed=args.seed,
            refresh=args.refresh,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
