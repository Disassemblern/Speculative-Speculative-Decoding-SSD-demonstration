"""Tests verifying the lossless property of rejection sampling in speculative decoding."""

from __future__ import annotations

import math

import pytest
import torch
from scipy.stats import chisquare

from src.speculative import rejection_sample


VOCAB = 10
DEVICE = "cpu"


def _uniform_probs(vocab: int = VOCAB) -> torch.Tensor:
    """Return a uniform probability vector."""
    return torch.ones(vocab) / vocab


def _peaked_probs(peak_idx: int, vocab: int = VOCAB) -> torch.Tensor:
    """Return a probability vector with mass concentrated at ``peak_idx``."""
    p = torch.zeros(vocab)
    p[peak_idx] = 1.0
    return p


def _random_probs(vocab: int = VOCAB, seed: int = 0) -> torch.Tensor:
    """Return a reproducibly random probability vector."""
    torch.manual_seed(seed)
    raw = torch.rand(vocab)
    return raw / raw.sum()


def test_rejection_sample_all_accepted() -> None:
    """When draft_probs == target_probs, all K tokens are accepted in every trial.

    Runs 100 independent trials and asserts ``k == K`` each time. When the
    acceptance ratio is exactly 1 the uniform draw ``u ~ U(0,1)`` always
    satisfies ``u <= 1``.
    """
    K = 4
    vocab = VOCAB

    torch.manual_seed(42)
    draft_tokens = list(range(K))
    draft_probs_list = [_random_probs(vocab=vocab, seed=i) for i in range(K)]
    target_probs_list = [p.clone() for p in draft_probs_list]
    bonus_probs = _random_probs(vocab=vocab, seed=99)

    for trial in range(100):
        k, bonus_token, acc_dp, acc_tp = rejection_sample(
            draft_tokens, draft_probs_list, target_probs_list, bonus_probs
        )
        assert k == K, f"Trial {trial}: expected k={K}, got k={k}"
        assert len(acc_dp) == K
        assert len(acc_tp) == K


def test_rejection_sample_first_rejected() -> None:
    """When target probability at draft_tokens[0] is 0, the first token is always rejected.

    With target probability zero at the chosen token, the acceptance ratio is
    0 / p_draft = 0, so the uniform sample never satisfies the acceptance
    condition, and k must be 0 on every trial.
    """
    K = 4
    vocab = VOCAB

    draft_tokens = [0] + list(range(1, K))

    draft_probs_list = [_random_probs(vocab=vocab, seed=i) for i in range(K)]
    draft_probs_list[0] = torch.zeros(vocab)
    draft_probs_list[0][0] = 1.0

    target_probs_list = [_random_probs(vocab=vocab, seed=i + 10) for i in range(K)]
    target_probs_list[0] = torch.zeros(vocab)

    bonus_probs = _random_probs(vocab=vocab, seed=99)

    for trial in range(100):
        k, bonus_token, acc_dp, acc_tp = rejection_sample(
            draft_tokens, draft_probs_list, target_probs_list, bonus_probs
        )
        assert k == 0, f"Trial {trial}: expected k=0, got k={k}"
        assert acc_dp == []
        assert acc_tp == []


def test_bonus_from_residual_distribution() -> None:
    """Bonus token distribution must match the normalised residual via chi-squared.

    Sets up a small vocabulary where the target and draft distributions differ,
    forces k=0 (immediate rejection at position 0), and checks that the
    empirical bonus-token distribution from 2000 samples matches the
    theoretical residual (p_target - p_draft).clamp(min=0) / norm.

    p-value threshold: 0.01 (generous to avoid flakiness from sampling noise).
    """
    vocab = VOCAB
    n_samples = 2000

    draft_token = 0
    draft_probs = torch.zeros(vocab)
    draft_probs[draft_token] = 1.0

    torch.manual_seed(7)
    raw = torch.rand(vocab)
    raw[draft_token] = 0.0
    target_probs = raw / raw.sum()

    K = 1
    draft_tokens = [draft_token]
    draft_probs_list = [draft_probs.clone()]
    target_probs_list = [target_probs.clone()]
    bonus_probs = torch.ones(vocab) / vocab

    residual = (target_probs - draft_probs).clamp(min=0.0)
    residual_norm = residual / residual.sum()
    expected_counts = residual_norm.numpy() * n_samples

    observed = torch.zeros(vocab, dtype=torch.int64)
    for _ in range(n_samples):
        k, bt, _, _ = rejection_sample(draft_tokens, draft_probs_list, target_probs_list, bonus_probs)
        assert k == 0, "Expected immediate rejection at position 0"
        observed[bt] += 1

    obs_np = observed.numpy().astype(float)
    exp_np = expected_counts.astype(float)

    nonzero_mask = exp_np > 0
    obs_nonzero = obs_np[nonzero_mask]
    exp_nonzero = exp_np[nonzero_mask]
    exp_nonzero = exp_nonzero / exp_nonzero.sum() * obs_nonzero.sum()

    _, p_value = chisquare(obs_nonzero, f_exp=exp_nonzero)
    assert p_value > 0.01, (
        f"Bonus token distribution mismatch: chi-squared p-value={p_value:.4f} < 0.01. "
        "Rejection sampling may not preserve the target distribution."
    )
