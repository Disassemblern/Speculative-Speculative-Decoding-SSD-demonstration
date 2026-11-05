"""Performance regression test: SSD must not be pathologically slow vs AR.

Catches the class of bug where SSD overhead dwarfs its parallelism benefit
(e.g., serialised cache build, K-extra-forwards on cache hit, oversized
cache budget). Runs on CPU with tiny GPT-2 models so total wall time stays
well under a minute.
"""

from __future__ import annotations

import time

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.autoregressive import generate_autoregressive
from src.ssd import SSDConfig, generate_ssd


_PREFERRED_MODELS = ("distilgpt2", "sshleifer/tiny-gpt2", "gpt2")


def _try_load_pair() -> tuple[object, object, object] | None:
    """Attempt to load a tiny CPU model pair from the preferred candidates.

    Tries each candidate id in order. The first one that loads is used for
    BOTH draft and target. The test skips when none are reachable (offline
    environment without any HF cache hit).

    Returns:
        Tuple of (draft_model, target_model, tokenizer) on success, or
        ``None`` when no candidate could be loaded.
    """
    for name in _PREFERRED_MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            draft = AutoModelForCausalLM.from_pretrained(name).eval()
            target = AutoModelForCausalLM.from_pretrained(name).eval()
            return draft, target, tok
        except Exception:  # noqa: BLE001
            continue
    return None


@pytest.fixture(scope="module")
def models() -> tuple[object, object, object]:
    """Load a tiny GPT-2 pair once per module, or skip when unavailable.

    Returns:
        Tuple of (draft_model, target_model, tokenizer).
    """
    pair = _try_load_pair()
    if pair is None:
        pytest.skip(
            "No tiny GPT-2 model available locally; skipping SSD perf test."
        )
    return pair


def _time_strategy(fn) -> float:
    """Measure wall-clock seconds for a zero-argument callable.

    Args:
        fn: Callable invoked once; return value is discarded.

    Returns:
        Elapsed seconds as a float.
    """
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def test_ssd_not_pathologically_slower_than_ar(
    models: tuple[object, object, object],
) -> None:
    """SSD wall-time must not be catastrophically slower than AR on CPU.

    On real GPU hardware SSD should be FASTER than AR (its whole point).
    On CPU, cache construction and the target verifier share the same
    cores, so the parallelism benefit disappears and the test only catches
    the structural class of bug where SSD spends an order of magnitude
    more time than AR.

    Threshold note: ``cache_budget_B=2`` is the minimum that exercises
    BOTH cache build phases. The 7x ceiling reflects the post-optimisation
    regime where each cache entry costs K single-token draft forwards on
    top of a SHARED primed KV cache instead of K full-context forwards.
    Measured CPU baseline after the optimisation is ~6.3x on distilgpt2;
    the earlier (pre-optimisation) baseline was ~12x and the previous
    ceiling was 8x.
    """
    draft, target, tok = models
    prompt = "Hello world"
    max_new = 16

    # Warmup pass to absorb model-init / cache-line / allocator overhead.
    torch.manual_seed(0)
    generate_autoregressive(
        target, tok, prompt, 2, temperature=1.0, device="cpu"
    )

    torch.manual_seed(0)
    t_ar = _time_strategy(
        lambda: generate_autoregressive(
            target,
            tok,
            prompt,
            max_new,
            temperature=1.0,
            device="cpu",
        )
    )

    torch.manual_seed(0)
    t_ssd = _time_strategy(
        lambda: generate_ssd(
            draft,
            target,
            tok,
            prompt,
            max_new,
            SSDConfig(K=4, temperature=1.0, device="cpu", cache_budget_B=2),
        )
    )

    assert t_ssd < 7.0 * t_ar, (
        f"SSD took {t_ssd:.2f}s vs AR {t_ar:.2f}s "
        f"(ratio {t_ssd / t_ar:.2f}). SSD has structural overhead bug."
    )
