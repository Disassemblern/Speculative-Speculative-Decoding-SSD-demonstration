"""Render summary plots from a benchmark run directory.

Reads ``summary.csv`` and writes two PNGs to the output directory:
    fig4_tokens_per_sec.png        - grouped bar chart by dataset x strategy
    fig7_acceptance_and_cache.png  - SSD acceptance + cache hit rate per dataset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


LOGGER = logging.getLogger("plot")


_STRATEGY_ORDER = ("ar", "sd", "ssd")
_STRATEGY_LABEL = {"ar": "AR", "sd": "SD", "ssd": "SSD"}


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the plotting script.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", type=Path, required=True)
    parser.add_argument("--out", dest="out_dir", type=Path, required=True)
    return parser.parse_args(argv)


def plot_tokens_per_sec(df: pd.DataFrame, out_path: Path) -> None:
    """Render the tokens/sec grouped bar chart.

    Args:
        df: Summary DataFrame containing ``tokens_per_sec_mean`` and
            ``tokens_per_sec_std`` columns indexed by strategy/dataset.
        out_path: Destination PNG path.

    Returns:
        None.
    """
    datasets = sorted(df["dataset"].unique().tolist())
    strategies = [s for s in _STRATEGY_ORDER if s in df["strategy"].unique()]
    width = 0.8 / max(len(strategies), 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(datasets))
    for i, strategy in enumerate(strategies):
        means: list[float] = []
        stds: list[float] = []
        for ds in datasets:
            row = df[(df["strategy"] == strategy) & (df["dataset"] == ds)]
            means.append(float(row["tokens_per_sec_mean"].iloc[0]) if len(row) else 0.0)
            stds.append(
                float(row["tokens_per_sec_std"].iloc[0])
                if len(row) and not pd.isna(row["tokens_per_sec_std"].iloc[0])
                else 0.0
            )
        offset = (i - (len(strategies) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=_STRATEGY_LABEL[strategy],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput by strategy and dataset")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ssd_acceptance_and_cache(df: pd.DataFrame, out_path: Path) -> None:
    """Render SSD-only acceptance rate and cache hit rate per dataset.

    Args:
        df: Summary DataFrame.
        out_path: Destination PNG path.

    Returns:
        None.
    """
    ssd = df[df["strategy"] == "ssd"]
    if ssd.empty:
        LOGGER.warning("no SSD rows in summary, skipping figure 7")
        # Always emit a placeholder so downstream tooling has a file.
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.text(0.5, 0.5, "No SSD data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    datasets = sorted(ssd["dataset"].unique().tolist())
    acc = [
        float(ssd[ssd["dataset"] == d]["acceptance_rate_mean"].iloc[0])
        for d in datasets
    ]
    hits_raw = [
        ssd[ssd["dataset"] == d]["cache_hit_rate_mean"].iloc[0] for d in datasets
    ]
    hits = [0.0 if pd.isna(v) else float(v) for v in hits_raw]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.bar(x - width / 2, acc, width=width, label="Acceptance rate")
    ax.bar(x + width / 2, hits, width=width, label="Cache hit rate")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("SSD acceptance and cache hit rate by dataset")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the plotting script.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    summary_path = args.in_dir / "summary.csv"
    if not summary_path.exists():
        LOGGER.error("missing summary.csv at %s", summary_path)
        return 1

    df = pd.read_csv(summary_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_tokens_per_sec(df, args.out_dir / "fig4_tokens_per_sec.png")
    plot_ssd_acceptance_and_cache(df, args.out_dir / "fig7_acceptance_and_cache.png")
    LOGGER.info("wrote plots to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
