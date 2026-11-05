"""Run the AR / SD / SSD benchmark across multiple datasets and write results.

Outputs per run go under ``--out-dir/run_<UTC-ISO>/``:
    raw.jsonl     - one row per (strategy, dataset, prompt)
    summary.csv   - aggregated mean+std per (strategy, dataset)
    metadata.json - git sha, nvidia-smi, library versions, CLI args, config
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Ensure the repository root is on sys.path when this script is invoked
# directly (so ``import src.*`` works without an installed package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from tqdm import tqdm

from scripts._bench_lib import (
    BenchConfig,
    load_config,
    load_prompts,
    make_run_dir,
    write_metadata,
)


LOGGER = logging.getLogger("benchmark")


DEFAULT_STRATEGIES = ("ar", "sd", "ssd")
DEFAULT_DATASETS = ("humaneval", "alpaca", "ultrafeedback", "gsm8k")


def _parse_csv(value: str) -> list[str]:
    """Split a comma-separated CLI value into trimmed lowercase tokens.

    Args:
        value: Raw CLI string.

    Returns:
        List of non-empty lowercase tokens.
    """
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the benchmark runner.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--strategies",
        type=_parse_csv,
        default=list(DEFAULT_STRATEGIES),
        help="Comma list of: ar, sd, ssd",
    )
    parser.add_argument(
        "--datasets",
        type=_parse_csv,
        default=list(DEFAULT_DATASETS),
        help="Comma list of: humaneval, alpaca, ultrafeedback, gsm8k",
    )
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Master RNG seed. Propagated to torch.manual_seed at startup, "
            "and to torch.manual_seed(seed + prompt_id) before each prompt "
            "so individual prompts are independently reproducible. Re-running "
            "with the same seed produces identical output_text byte-for-byte."
        ),
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=2048,
        help=(
            "Maximum prompt length in tokens. Prompts longer than this are "
            "truncated to the last N tokens (recent context retained, prefix "
            "dropped) to bound target-model VRAM."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _plan(
    cfg: BenchConfig,
    args: argparse.Namespace,
) -> tuple[list[str], list[str], int, int, float, int, int]:
    """Resolve effective strategies, datasets, and default hyper-parameters.

    Args:
        cfg: Parsed configuration.
        args: Parsed CLI namespace.

    Returns:
        Tuple of (strategies, datasets, K, max_new_tokens, temperature,
        num_prompts, cache_budget_B).
    """
    strategies = [s for s in args.strategies if s in {"ar", "sd", "ssd"}]
    datasets = [d for d in args.datasets if d in set(DEFAULT_DATASETS)]
    K = args.K if args.K is not None else cfg.K
    max_new_tokens = (
        args.max_new_tokens if args.max_new_tokens is not None else cfg.max_new_tokens
    )
    temperature = (
        args.temperature if args.temperature is not None else cfg.temperature
    )
    num_prompts = args.num_prompts if args.num_prompts is not None else cfg.num_prompts
    cache_budget_B = cfg.cache_budget_B
    return (
        strategies,
        datasets,
        K,
        max_new_tokens,
        temperature,
        num_prompts,
        cache_budget_B,
    )


def _truncate_prompt(
    prompt: str,
    tokenizer: Any,
    max_input_tokens: int,
) -> tuple[str, int, bool]:
    """Tokenize ``prompt`` and truncate to the last ``max_input_tokens`` tokens.

    Long prompts are truncated from the FRONT so the most recent context
    (closest to the generation cursor) is preserved.

    Args:
        prompt: Raw input text.
        tokenizer: HuggingFace tokenizer used for length measurement.
        max_input_tokens: Maximum allowed token length for the prompt.

    Returns:
        Tuple of (effective_prompt, original_token_count, truncated_flag).
    """
    ids = tokenizer(prompt, return_tensors=None, add_special_tokens=False)[
        "input_ids"
    ]
    orig_len = len(ids)
    if orig_len <= max_input_tokens:
        return prompt, orig_len, False
    kept = ids[-max_input_tokens:]
    truncated_text = tokenizer.decode(kept, skip_special_tokens=True)
    return truncated_text, orig_len, True


def _run_one(
    strategy: str,
    prompt: str,
    *,
    draft_model: Any,
    target_model: Any,
    tokenizer: Any,
    K: int,
    max_new_tokens: int,
    temperature: float,
    target_device: str,
    draft_device: str,
    cache_budget_B: int = 8,
) -> dict[str, Any]:
    """Run a single (strategy, prompt) pair and return its metrics row.

    The outer wall-clock timer is bracketed by ``sync_if_cuda`` so that
    asynchronous CUDA launches are flushed before ``time.perf_counter`` is
    sampled. Inner per-round metrics inside the strategies are already
    synced individually.

    Args:
        strategy: One of ``"ar"``, ``"sd"``, ``"ssd"``.
        prompt: Input text.
        draft_model: Loaded draft model.
        target_model: Loaded target model.
        tokenizer: Shared tokenizer.
        K: Speculative draft length.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        target_device: Device string for the target model (used by AR).
        draft_device: Device string for the draft model.

    Returns:
        Dict with timing, throughput, and strategy-specific fields.
    """
    from src.autoregressive import generate_autoregressive
    from src.speculative import SpeculativeConfig, generate_speculative
    from src.ssd import SSDConfig, generate_ssd
    from src.utils import sync_if_cuda

    sync_if_cuda(draft_device)
    sync_if_cuda(target_device)
    t0 = time.perf_counter()
    if strategy == "ar":
        text, metrics = generate_autoregressive(
            model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=target_device,
        )
    elif strategy == "sd":
        text, metrics = generate_speculative(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            config=SpeculativeConfig(
                K=K, temperature=temperature, device=target_device
            ),
        )
    elif strategy == "ssd":
        text, metrics = generate_ssd(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            config=SSDConfig(
                K=K,
                temperature=temperature,
                device=target_device,
                cache_budget_B=cache_budget_B,
            ),
        )
    else:  # pragma: no cover - filtered upstream
        raise ValueError(f"unknown strategy: {strategy!r}")
    sync_if_cuda(draft_device)
    sync_if_cuda(target_device)
    wall = time.perf_counter() - t0

    return {
        "tokens_generated": metrics.total_tokens,
        "wall_time_s": wall,
        "tokens_per_sec": metrics.tokens_per_second,
        "acceptance_rate": metrics.mean_acceptance_rate,
        "cache_hit_rate": metrics.cache_hit_rate,
        "output_text": text[:512],
    }


def _warmup_strategy(
    strategy: str,
    *,
    draft_model: Any,
    target_model: Any,
    tokenizer: Any,
    K: int,
    temperature: float,
    target_device: str,
    draft_device: str,
    cache_budget_B: int = 8,
) -> None:
    """Run a tiny generation once per strategy to absorb CUDA init / JIT.

    The output is discarded. Without this, the first real prompt's wall
    time would be inflated by context creation, kernel compilation, and
    autotuner runs.

    Args:
        strategy: One of ``"ar"``, ``"sd"``, ``"ssd"``.
        draft_model: Loaded draft model.
        target_model: Loaded target model.
        tokenizer: Shared tokenizer.
        K: Speculative draft length used at real-run time.
        temperature: Sampling temperature used at real-run time.
        target_device: Device string for the target model.
        draft_device: Device string for the draft model.

    Returns:
        None.
    """
    try:
        _run_one(
            strategy=strategy,
            prompt="Hello",
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            K=K,
            max_new_tokens=8,
            temperature=temperature,
            target_device=target_device,
            draft_device=draft_device,
            cache_budget_B=cache_budget_B,
        )
        LOGGER.info("warmup complete: strategy=%s", strategy)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("warmup failed for strategy=%s: %s", strategy, exc)


def _aggregate(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate raw rows into per (strategy, dataset) mean/std stats.

    Args:
        rows: List of raw result rows.

    Returns:
        DataFrame with mean and std of key numeric metrics.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    numeric = [
        "tokens_generated",
        "wall_time_s",
        "tokens_per_sec",
        "acceptance_rate",
        "cache_hit_rate",
    ]
    grouped = df.groupby(["strategy", "dataset"])[numeric].agg(["mean", "std"])
    grouped.columns = [f"{c}_{stat}" for c, stat in grouped.columns]
    return grouped.reset_index()


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the benchmark runner.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    cfg = load_config(args.config)
    (
        strategies,
        datasets,
        K,
        max_new_tokens,
        temperature,
        num_prompts,
        cache_budget_B,
    ) = _plan(cfg, args)

    plan_summary = {
        "strategies": strategies,
        "datasets": datasets,
        "K": K,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "num_prompts": num_prompts,
        "cache_budget_B": cache_budget_B,
        "draft": cfg.draft.name,
        "target": cfg.target.name,
        "draft_device": cfg.draft.device,
        "target_device": cfg.target.device,
    }
    LOGGER.info("plan: %s", json.dumps(plan_summary, indent=2))

    if args.dry_run:
        LOGGER.info("dry-run, exiting")
        return 0

    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from src.utils import load_models

    draft_model, target_model, tokenizer = load_models(
        draft_name=cfg.draft.name,
        target_name=cfg.target.name,
        draft_device=cfg.draft.device,
        target_device=cfg.target.device,
        dtype=cfg.target.dtype,
    )

    run_dir = make_run_dir(args.out_dir)
    raw_path = run_dir / "raw.jsonl"

    # Write LATEST pointer immediately so partial-failure runs are still
    # discoverable by downstream tooling (run_all.sh, plot scripts).
    latest_path = args.out_dir / "LATEST"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir), encoding="utf-8")

    extra_meta = vars(args) | plan_summary | {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    write_metadata(run_dir, cfg, extra_meta)

    # Warmup once per strategy (model is the same across datasets).
    for strategy in strategies:
        _warmup_strategy(
            strategy=strategy,
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            K=K,
            temperature=temperature,
            target_device=cfg.target.device,
            draft_device=cfg.draft.device,
            cache_budget_B=cache_budget_B,
        )

    raw_rows: list[dict[str, Any]] = []
    total = sum(
        len(load_prompts(args.data_dir, ds, num_prompts)) * len(strategies)
        for ds in datasets
    )
    pbar = tqdm(total=total, desc="bench", unit="run")

    with raw_path.open("w", encoding="utf-8") as raw_fh:
        for dataset in datasets:
            prompts = load_prompts(args.data_dir, dataset, num_prompts)
            if not prompts:
                LOGGER.warning("no prompts for dataset=%s, skipping", dataset)
                continue
            for strategy in strategies:
                for row in prompts:
                    prompt_id_val = row["id"]
                    # Derive a deterministic per-prompt seed offset so a
                    # single prompt can be replayed independently.
                    try:
                        seed_offset = int(prompt_id_val)
                    except (TypeError, ValueError):
                        seed_offset = abs(hash(str(prompt_id_val))) % (2**31)
                    prompt_seed = args.seed + seed_offset
                    torch.manual_seed(prompt_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(prompt_seed)

                    eff_prompt, input_tok_count, truncated = _truncate_prompt(
                        row["prompt"], tokenizer, args.max_input_tokens
                    )

                    try:
                        result = _run_one(
                            strategy=strategy,
                            prompt=eff_prompt,
                            draft_model=draft_model,
                            target_model=target_model,
                            tokenizer=tokenizer,
                            K=K,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            target_device=cfg.target.device,
                            draft_device=cfg.draft.device,
                            cache_budget_B=cache_budget_B,
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.error(
                            "failed: strategy=%s dataset=%s id=%s: %s",
                            strategy,
                            dataset,
                            prompt_id_val,
                            exc,
                        )
                        pbar.update(1)
                        continue
                    full = {
                        "strategy": strategy,
                        "dataset": dataset,
                        "prompt_id": prompt_id_val,
                        "seed": prompt_seed,
                        "input_token_count": input_tok_count,
                        "truncated": truncated,
                        **result,
                    }
                    raw_rows.append(full)
                    raw_fh.write(json.dumps(full, ensure_ascii=False) + "\n")
                    raw_fh.flush()
                    pbar.update(1)
    pbar.close()

    summary = _aggregate(raw_rows)
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("wrote %d raw rows to %s", len(raw_rows), raw_path)
    LOGGER.info("wrote summary to %s", summary_path)
    LOGGER.info("run dir: %s", run_dir)
    # Re-affirm LATEST on success and print the run dir as the final stdout
    # line so callers can capture it deterministically.
    latest_path.write_text(str(run_dir), encoding="utf-8")
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
