"""Speculative decoding: draft K tokens with a small model, verify with a large model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.metrics import MetricsAccumulator, RoundResult, SessionMetrics
from src.utils import decode, encode, logits_to_probs, sample_token


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Attributes:
        K: Number of draft tokens to speculate per round.
        temperature: Sampling temperature for both draft and target.
        device: Torch device string.
    """

    K: int = 4
    temperature: float = 1.0
    device: str = "cuda"


def draft_K_tokens(
    draft_model: PreTrainedModel,
    input_ids: Tensor,
    K: int,
    temperature: float,
    past_key_values: object | None,
) -> tuple[list[int], list[Tensor], object]:
    """Autoregressively sample K draft tokens from the draft model.

    Args:
        draft_model: The smaller draft language model.
        input_ids: Context token ids of shape ``(1, seq_len)``.
        K: Number of tokens to draft.
        temperature: Sampling temperature.
        past_key_values: KV cache from a previous call, or ``None``.

    Returns:
        Tuple of (draft_token_ids, draft_prob_tensors, updated_kv_cache).
        ``draft_prob_tensors[i]`` is the full probability vector ``(vocab,)``
        from which ``draft_token_ids[i]`` was sampled.
    """
    draft_tokens: list[int] = []
    draft_probs: list[Tensor] = []
    current_ids = input_ids

    with torch.no_grad():
        for _ in range(K):
            outputs = draft_model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values

            probs = logits_to_probs(logits, temperature)
            token_id: int = torch.multinomial(probs, num_samples=1).item()

            draft_tokens.append(token_id)
            draft_probs.append(probs)
            current_ids = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)

    return draft_tokens, draft_probs, past_key_values


def verify_draft_tokens(
    target_model: PreTrainedModel,
    input_ids: Tensor,
    draft_tokens: list[int],
    temperature: float,
) -> tuple[list[Tensor], Tensor]:
    """Run one forward pass on context + draft tokens to get target distributions.

    The target distribution for ``draft_tokens[i]`` lives at logit index
    ``context_len - 1 + i`` of the concatenated input (0-based). The bonus
    distribution is at index ``context_len - 1 + K``.

    Args:
        target_model: The larger target language model.
        input_ids: Context token ids of shape ``(1, context_len)``.
        draft_tokens: List of K draft token ids.
        temperature: Sampling temperature.

    Returns:
        Tuple of (target_probs_list, bonus_probs).
        ``target_probs_list[i]`` is shape ``(vocab,)``; ``bonus_probs`` is
        shape ``(vocab,)``.
    """
    K = len(draft_tokens)
    context_len = input_ids.size(1)
    device = input_ids.device

    draft_tensor = torch.tensor(draft_tokens, dtype=torch.long, device=device).unsqueeze(0)
    full_input = torch.cat([input_ids, draft_tensor], dim=1)

    with torch.no_grad():
        outputs = target_model(input_ids=full_input, use_cache=False)

    logits = outputs.logits[0]

    target_probs_list: list[Tensor] = []
    for i in range(K):
        pos = context_len - 1 + i
        probs = logits_to_probs(logits[pos], temperature)
        target_probs_list.append(probs)

    bonus_probs = logits_to_probs(logits[context_len - 1 + K], temperature)
    return target_probs_list, bonus_probs


def rejection_sample(
    draft_tokens: list[int],
    draft_probs_list: list[Tensor],
    target_probs_list: list[Tensor],
    bonus_probs: Tensor,
) -> tuple[int, int, list[float], list[float]]:
    """Apply speculative decoding rejection sampling.

    For each draft token ``t_i``, accept with probability
    ``min(1, p_target(t_i) / p_draft(t_i))``. On rejection at position ``i``,
    sample the bonus token from the normalised residual
    ``(p_target - p_draft).clamp(min=0)``. If all K tokens are accepted, sample
    the bonus token from the target bonus distribution.

    Args:
        draft_tokens: List of K draft token ids.
        draft_probs_list: Draft probability tensors, one per draft token.
        target_probs_list: Target probability tensors, one per draft token.
        bonus_probs: Target probability tensor for the position after all K drafts.

    Returns:
        Tuple of (k, bonus_token, accepted_draft_probs, accepted_target_probs)
        where ``k`` is the number of accepted draft tokens (0 … K).
    """
    K = len(draft_tokens)
    accepted_draft_probs: list[float] = []
    accepted_target_probs: list[float] = []

    for i in range(K):
        t_i = draft_tokens[i]
        p_d = draft_probs_list[i][t_i].item()
        p_t = target_probs_list[i][t_i].item()

        ratio = p_t / p_d if p_d > 0.0 else 1.0
        u = torch.rand(1).item()

        if u <= min(1.0, ratio):
            accepted_draft_probs.append(p_d)
            accepted_target_probs.append(p_t)
        else:
            residual = (target_probs_list[i] - draft_probs_list[i]).clamp(min=0.0)
            residual_sum = residual.sum()
            if residual_sum > 0:
                residual = residual / residual_sum
            else:
                residual = torch.ones_like(residual) / residual.size(0)
            bonus_token: int = torch.multinomial(residual, num_samples=1).item()
            return i, bonus_token, accepted_draft_probs, accepted_target_probs

    bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()
    return K, bonus_token, accepted_draft_probs, accepted_target_probs


def generate_speculative(
    draft_model: PreTrainedModel,
    target_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int,
    config: SpeculativeConfig,
) -> tuple[str, SessionMetrics]:
    """Generate text using speculative decoding.

    Each round drafts K tokens with the draft model, verifies them with one
    target forward pass, and applies rejection sampling. No KV cache is shared
    across rounds for the target model; the draft KV cache is reset each round
    because the bonus token is only known after rejection sampling.

    Args:
        draft_model: The smaller draft language model.
        target_model: The larger target language model.
        tokenizer: Tokenizer shared by both models.
        prompt: Input text to condition on.
        max_new_tokens: Maximum number of tokens to generate.
        config: Speculative decoding hyper-parameters.

    Returns:
        Tuple of (generated_text, session_metrics).
    """
    input_ids = encode(prompt, tokenizer, config.device)
    accumulator = MetricsAccumulator()
    generated: list[int] = []

    while len(generated) < max_new_tokens:
        context = torch.cat(
            [input_ids, torch.tensor([generated], dtype=torch.long, device=config.device)],
            dim=1,
        ) if generated else input_ids

        remaining = max_new_tokens - len(generated)
        K = min(config.K, remaining)

        t_start = time.perf_counter()

        draft_tokens, draft_probs_list, _ = draft_K_tokens(
            draft_model, context, K, config.temperature, past_key_values=None
        )
        target_probs_list, bonus_probs = verify_draft_tokens(
            target_model, context, draft_tokens, config.temperature
        )
        k, bonus_token, acc_dp, acc_tp = rejection_sample(
            draft_tokens, draft_probs_list, target_probs_list, bonus_probs
        )

        wall_time = time.perf_counter() - t_start

        accepted = draft_tokens[:k]
        round_tokens = accepted + [bonus_token]
        generated.extend(round_tokens)

        result = RoundResult(
            tokens_accepted=k,
            bonus_token=bonus_token,
            all_tokens=round_tokens,
            wall_time_s=wall_time,
            cache_hit=None,
            draft_probs=acc_dp,
            target_probs=acc_tp,
        )
        accumulator.record(result)

        if bonus_token == tokenizer.eos_token_id:
            break

    text = decode(generated, tokenizer)
    metrics = accumulator.finalize(K=config.K)
    return text, metrics
