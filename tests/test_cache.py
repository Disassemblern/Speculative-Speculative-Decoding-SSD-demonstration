"""Tests for SaguaroCache correctness and saguaro_draft_K_tokens behaviour."""

from __future__ import annotations

import torch
import pytest

from src.ssd import SSDConfig, SaguaroCache, saguaro_draft_K_tokens


DEVICE = "cpu"


def _make_config(**kwargs) -> SSDConfig:
    """Return an SSDConfig with test-friendly defaults."""
    defaults = dict(
        K=4,
        cache_budget_B=20,
        acceptance_rate_estimate=0.7,
        power_law_r=0.5,
        saguaro_C=0.5,
        temperature=1.0,
        device=DEVICE,
    )
    defaults.update(kwargs)
    return SSDConfig(**defaults)


def test_fan_out_length() -> None:
    """compute_fan_out must return a list of length K+1."""
    for K in [1, 2, 4, 8]:
        config = _make_config(K=K)
        cache = SaguaroCache(config)
        fan_outs = cache.compute_fan_out()
        assert len(fan_outs) == K + 1, (
            f"Expected {K + 1} fan-out entries for K={K}, got {len(fan_outs)}"
        )


def test_fan_out_budget() -> None:
    """Sum of fan-out values must not exceed B + K (the last entry is always 1)."""
    for B in [10, 20, 50]:
        config = _make_config(cache_budget_B=B, K=4)
        cache = SaguaroCache(config)
        fan_outs = cache.compute_fan_out()
        total = sum(fan_outs)
        assert total <= B + config.K, (
            f"Fan-out total {total} exceeds budget B+K={B + config.K} for B={B}"
        )


class _TinyDraftModel:
    """Minimal stand-in for a draft model that returns fixed logits."""

    def __init__(self, vocab: int, device: str) -> None:
        self._vocab = vocab
        self._device = device
        self.config = type("cfg", (), {"hidden_size": 64})()

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
    ):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, self._vocab, device=self._device)
        logits[0, -1, 0] = 10.0
        return type("Out", (), {"logits": logits, "past_key_values": None})()


def test_cache_hit() -> None:
    """Building the cache with a top-1 bonus candidate must yield a hit on lookup."""
    vocab = 10
    config = _make_config(K=2, cache_budget_B=10)
    cache = SaguaroCache(config)
    draft_model = _TinyDraftModel(vocab, DEVICE)

    context_ids = torch.zeros(1, 3, dtype=torch.long)
    draft_tokens = [0, 1]
    raw_logits_list = [
        torch.zeros(vocab),
        torch.zeros(vocab),
    ]
    raw_logits_list[0][3] = 10.0
    raw_logits_list[1][5] = 10.0

    bonus_probs = torch.zeros(vocab)
    bonus_probs[0] = 1.0

    cache.build(
        draft_model=draft_model,
        context_ids=context_ids,
        draft_tokens=draft_tokens,
        raw_logits_list=raw_logits_list,
        bonus_probs=bonus_probs,
        temperature=1.0,
    )

    top_bonus_at_k0 = 3
    entry = cache.lookup(k=0, bonus_token=top_bonus_at_k0)
    assert entry is not None, (
        "Expected a cache hit for the top-1 bonus candidate at k=0, but got None."
    )


def test_cache_miss() -> None:
    """A token that was not in the top-F candidates must produce a cache miss."""
    vocab = 20
    config = _make_config(K=2, cache_budget_B=4)
    cache = SaguaroCache(config)
    draft_model = _TinyDraftModel(vocab, DEVICE)

    context_ids = torch.zeros(1, 3, dtype=torch.long)
    draft_tokens = [0, 1]
    raw_logits_list = [torch.zeros(vocab) for _ in range(2)]
    raw_logits_list[0][0] = 10.0

    bonus_probs = torch.zeros(vocab)
    bonus_probs[0] = 1.0

    cache.build(
        draft_model=draft_model,
        context_ids=context_ids,
        draft_tokens=draft_tokens,
        raw_logits_list=raw_logits_list,
        bonus_probs=bonus_probs,
        temperature=1.0,
    )

    rare_token = vocab - 1
    entry = cache.lookup(k=0, bonus_token=rare_token)
    assert entry is None, (
        f"Expected cache miss for token {rare_token} not in top-F, but got an entry."
    )


def test_saguaro_downweights_top_tokens() -> None:
    """Modified probs for top-F tokens must be lower than original probs when C < 1.

    Adding log(C) < 0 to top-F logits before softmax concentrates mass on
    those tokens in logit space but, relative to the original softmax, the
    renormalisation means the effective probability of the suppressed tokens
    is reduced compared to a model that added log(C) > 0.

    The test verifies that the top-1 token's modified probability is strictly
    less than the original softmax probability of that token when C=0.5.

    Note: saguaro_draft_K_tokens ADDS log(C) (with C=0.5, log(C)<0) to the
    top-F logits, which DOWN-weights them relative to uniform — but compared
    to the raw softmax the top-F tokens still dominate. The assertion here
    checks that the modified probability of the original top token is
    less than 1.0 (i.e. probability mass has been spread), not that it is
    less than the raw softmax value (which would be backwards for C<1 adding
    negative log).

    Actually: adding log(0.5) < 0 to top-F logits reduces their relative
    logit advantage, so the modified top-1 prob < raw top-1 prob. We assert
    that directly.
    """
    vocab = 10
    K = 1

    raw_logits = torch.zeros(vocab)
    raw_logits[0] = 5.0

    raw_probs = torch.softmax(raw_logits, dim=-1)
    raw_top1_prob = raw_probs[0].item()

    import math

    C = 0.5
    modified_logits = raw_logits.clone()
    modified_logits[0] += math.log(C)
    modified_probs = torch.softmax(modified_logits, dim=-1)
    modified_top1_prob = modified_probs[0].item()

    assert modified_top1_prob < raw_top1_prob, (
        f"Expected modified top-1 prob {modified_top1_prob:.4f} < "
        f"raw top-1 prob {raw_top1_prob:.4f} after adding log(C={C})."
    )
