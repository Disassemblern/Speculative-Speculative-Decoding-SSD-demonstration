"""Speculative Speculative Decoding (SSD) with Saguaro cache pre-computation."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.metrics import MetricsAccumulator, RoundResult, SessionMetrics
from src.speculative import rejection_sample, verify_draft_tokens
from src.utils import decode, encode, logits_to_probs, top_k_token_ids


@dataclass
class SSDConfig:
    """Configuration for Speculative Speculative Decoding.

    Attributes:
        K: Number of draft tokens per round.
        temperature: Sampling temperature for both models.
        device: Torch device string.
        cache_budget_B: Total number of cache entries to pre-fill.
        saguaro_C: Concentration factor for top-F token upweighting (0 < C <= 1).
        acceptance_rate_estimate: Prior estimate of draft acceptance rate.
        power_law_r: Power-law exponent for Theorem 12 fan-out weights.
        fallback: Strategy when cache misses; ``"neural"`` re-runs draft model.
    """

    K: int = 4
    temperature: float = 1.0
    device: str = "cuda"
    cache_budget_B: int = 20
    saguaro_C: float = 0.5
    acceptance_rate_estimate: float = 0.7
    power_law_r: float = 0.5
    fallback: str = "neural"


@dataclass
class CacheEntry:
    """A cached speculative draft ready for the next round.

    Attributes:
        k: Acceptance count that produced this entry.
        bonus_token: The bonus token that preceded this entry's context.
        next_draft_tokens: Pre-computed draft token ids for the next round.
        next_draft_probs: Pre-computed draft probability tensors for the next round.
    """

    k: int
    bonus_token: int
    next_draft_tokens: list[int]
    next_draft_probs: list[Tensor]


class SaguaroCache:
    """Pre-computes and stores draft tokens for all likely next contexts.

    Uses a geometric fan-out strategy (Theorem 12) to allocate the cache
    budget across the K rejection positions plus the all-accepted case.
    """

    def __init__(self, config: SSDConfig) -> None:
        """Initialise an empty cache.

        Args:
            config: The SSD configuration holding budget and fan-out params.
        """
        self._config = config
        self._store: dict[tuple[int, int], CacheEntry] = {}

    def compute_fan_out(self) -> list[int]:
        """Compute per-position fan-out counts using Theorem 12 geometric weights.

        Allocates ``cache_budget_B`` entries across K+1 positions. Position k
        receives weight proportional to ``a_p^(k/(1+r))``, where ``a_p`` is
        the acceptance rate estimate and ``r`` is the power-law exponent.
        Position K (all-accepted bonus) always has fan-out 1.

        Returns:
            List of K+1 fan-out values ``[F_0, F_1, ..., F_{K-1}, 1]``.
        """
        K = self._config.K
        a_p = self._config.acceptance_rate_estimate
        r = self._config.power_law_r
        B = self._config.cache_budget_B

        weights = [a_p ** (k / (1.0 + r)) for k in range(K)]
        weight_sum = sum(weights)

        if weight_sum > 0:
            F_0 = max(1, round((B - 1) / weight_sum))
        else:
            F_0 = 1

        fan_outs = [max(1, round(F_0 * w)) for w in weights]
        fan_outs.append(1)
        return fan_outs

    def build(
        self,
        draft_model: PreTrainedModel,
        context_ids: Tensor,
        draft_tokens: list[int],
        raw_logits_list: list[Tensor],
        bonus_probs: Tensor,
        temperature: float,
    ) -> None:
        """Pre-fill the cache for all likely next contexts.

        For each rejection position k in 0..K, iterates over the top-F_k
        bonus-token candidates at that position, constructs the hypothetical
        next context, and runs the draft model to get next-round draft tokens.

        Args:
            draft_model: The smaller draft language model.
            context_ids: Current context token ids of shape ``(1, context_len)``.
            draft_tokens: The K draft tokens used in the current round.
            raw_logits_list: Raw logit tensors (pre-temperature) at each of the
                K draft positions, used to select top-F_k candidates.
            bonus_probs: Target probability tensor for the all-accepted position.
            temperature: Sampling temperature.

        Returns:
            None.
        """
        self._store.clear()
        fan_outs = self.compute_fan_out()
        K = self._config.K
        device = context_ids.device

        for k in range(K + 1):
            F_k = fan_outs[k]

            if k < K:
                candidates = top_k_token_ids(raw_logits_list[k], F_k)
            else:
                candidates = top_k_token_ids(bonus_probs, F_k)

            prefix = list(draft_tokens[:k])
            for bonus_candidate in candidates:
                hyp_tokens = prefix + [bonus_candidate]
                hyp_tensor = torch.tensor(hyp_tokens, dtype=torch.long, device=device).unsqueeze(0)
                hyp_context = torch.cat([context_ids, hyp_tensor], dim=1)

                next_draft_tokens, next_draft_probs = _draft_K_tokens_no_cache(
                    draft_model, hyp_context, K, temperature
                )

                entry = CacheEntry(
                    k=k,
                    bonus_token=bonus_candidate,
                    next_draft_tokens=next_draft_tokens,
                    next_draft_probs=next_draft_probs,
                )
                self._store[(k, bonus_candidate)] = entry

    def lookup(self, k: int, bonus_token: int) -> Optional[CacheEntry]:
        """Retrieve a cached draft entry for a given (k, bonus_token) pair.

        Args:
            k: The acceptance count from rejection sampling.
            bonus_token: The bonus token id produced by rejection sampling.

        Returns:
            The :class:`CacheEntry` if it exists, otherwise ``None``.
        """
        return self._store.get((k, bonus_token), None)

    def clear(self) -> None:
        """Discard all cached entries.

        Returns:
            None.
        """
        self._store.clear()


def _draft_K_tokens_no_cache(
    draft_model: PreTrainedModel,
    input_ids: Tensor,
    K: int,
    temperature: float,
) -> tuple[list[int], list[Tensor]]:
    """Draft K tokens autoregressively without a persistent KV cache.

    Helper used inside cache build to avoid cache state bleeding between
    hypothetical contexts.

    Args:
        draft_model: The smaller draft language model.
        input_ids: Context tensor of shape ``(1, seq_len)``.
        K: Number of tokens to draft.
        temperature: Sampling temperature.

    Returns:
        Tuple of (token_ids, prob_tensors).
    """
    tokens: list[int] = []
    probs: list[Tensor] = []
    current_ids = input_ids
    past_kv = None

    with torch.no_grad():
        for _ in range(K):
            outputs = draft_model(
                input_ids=current_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            past_kv = outputs.past_key_values
            p = logits_to_probs(logits, temperature)
            t: int = torch.multinomial(p, num_samples=1).item()
            tokens.append(t)
            probs.append(p)
            current_ids = torch.tensor([[t]], dtype=torch.long, device=input_ids.device)

    return tokens, probs


def saguaro_draft_K_tokens(
    draft_model: PreTrainedModel,
    input_ids: Tensor,
    K: int,
    temperature: float,
    fan_outs: list[int],
    C: float,
) -> tuple[list[int], list[Tensor], list[Tensor]]:
    """Draft K tokens with Saguaro top-F concentration.

    At each step the raw logits are retrieved, then ``log(C)`` is added to the
    top-F_k entries before converting to probabilities and sampling. The
    modified probabilities are used in rejection sampling to maintain
    losslessness; the raw logits are returned separately for cache construction.

    Args:
        draft_model: The smaller draft language model.
        input_ids: Context tensor of shape ``(1, seq_len)``.
        K: Number of tokens to draft.
        temperature: Sampling temperature.
        fan_outs: Fan-out list from :meth:`SaguaroCache.compute_fan_out`; only
            the first K entries (positions 0..K-1) are used here.
        C: Concentration scalar in (0, 1]. ``log(C)`` is added to top-F logits.

    Returns:
        Tuple of (draft_tokens, modified_probs_list, raw_logits_list).
    """
    draft_tokens: list[int] = []
    modified_probs_list: list[Tensor] = []
    raw_logits_list: list[Tensor] = []
    current_ids = input_ids
    past_kv = None
    log_c = math.log(C)

    with torch.no_grad():
        for i in range(K):
            outputs = draft_model(
                input_ids=current_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            raw_logits = outputs.logits[0, -1, :].clone()
            past_kv = outputs.past_key_values
            raw_logits_list.append(raw_logits)

            F_k = fan_outs[i] if i < len(fan_outs) else 1
            modified_logits = raw_logits.clone()

            top_indices = torch.topk(raw_logits, k=min(F_k, raw_logits.size(0))).indices
            modified_logits[top_indices] += log_c

            probs = logits_to_probs(modified_logits, temperature)
            modified_probs_list.append(probs)

            token_id: int = torch.multinomial(probs, num_samples=1).item()
            draft_tokens.append(token_id)
            current_ids = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)

    return draft_tokens, modified_probs_list, raw_logits_list


def generate_ssd(
    draft_model: PreTrainedModel,
    target_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int,
    config: SSDConfig,
) -> tuple[str, SessionMetrics]:
    """Generate text using Speculative Speculative Decoding.

    Each round overlaps target verification with Saguaro cache construction
    using a ``ThreadPoolExecutor``. When the cache has an entry for the
    (k, bonus_token) outcome, the next round's draft is taken from cache;
    otherwise the draft model is re-run (neural fallback).

    Args:
        draft_model: The smaller draft language model.
        target_model: The larger target language model.
        tokenizer: Tokenizer shared by both models.
        prompt: Input text to condition on.
        max_new_tokens: Maximum number of tokens to generate.
        config: SSD hyper-parameters.

    Returns:
        Tuple of (generated_text, session_metrics).
    """
    input_ids = encode(prompt, tokenizer, config.device)
    accumulator = MetricsAccumulator()
    generated: list[int] = []
    cache = SaguaroCache(config)

    pending_draft_tokens: Optional[list[int]] = None
    pending_draft_probs: Optional[list[Tensor]] = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        while len(generated) < max_new_tokens:
            context = (
                torch.cat(
                    [
                        input_ids,
                        torch.tensor([generated], dtype=torch.long, device=config.device),
                    ],
                    dim=1,
                )
                if generated
                else input_ids
            )

            remaining = max_new_tokens - len(generated)
            K = min(config.K, remaining)
            fan_outs = cache.compute_fan_out()

            t_start = time.perf_counter()

            if pending_draft_tokens is not None and len(pending_draft_tokens) == K:
                draft_tokens = pending_draft_tokens
                modified_probs_list = pending_draft_probs
                pending_draft_tokens = None
                pending_draft_probs = None

                raw_logits_list: list[Tensor] = []
                tmp_ids = context
                tmp_past = None
                with torch.no_grad():
                    for i in range(K):
                        outputs = draft_model(
                            input_ids=tmp_ids,
                            past_key_values=tmp_past,
                            use_cache=True,
                        )
                        raw_logits_list.append(outputs.logits[0, -1, :].clone())
                        tmp_past = outputs.past_key_values
                        tmp_ids = torch.tensor(
                            [[draft_tokens[i]]], dtype=torch.long, device=config.device
                        )
            else:
                draft_tokens, modified_probs_list, raw_logits_list = saguaro_draft_K_tokens(
                    draft_model, context, K, config.temperature, fan_outs, config.saguaro_C
                )

            verifier_future = executor.submit(
                verify_draft_tokens,
                target_model,
                context,
                draft_tokens,
                config.temperature,
            )

            target_probs_list, bonus_probs = verifier_future.result()

            cache_future = executor.submit(
                cache.build,
                draft_model,
                context,
                draft_tokens,
                raw_logits_list,
                bonus_probs,
                config.temperature,
            )

            k, bonus_token, acc_dp, acc_tp = rejection_sample(
                draft_tokens, modified_probs_list, target_probs_list, bonus_probs
            )

            cache_future.result()

            wall_time = time.perf_counter() - t_start

            entry = cache.lookup(k, bonus_token)
            cache_hit: bool

            if entry is not None and len(entry.next_draft_tokens) == K:
                pending_draft_tokens = entry.next_draft_tokens
                pending_draft_probs = entry.next_draft_probs
                cache_hit = True
            else:
                cache_hit = False
                if config.fallback == "neural":
                    next_context_tokens = list(draft_tokens[:k]) + [bonus_token]
                    next_context = torch.cat(
                        [
                            context,
                            torch.tensor(
                                [next_context_tokens], dtype=torch.long, device=config.device
                            ),
                        ],
                        dim=1,
                    )
                    fb_tokens, fb_probs, _ = saguaro_draft_K_tokens(
                        draft_model,
                        next_context,
                        K,
                        config.temperature,
                        fan_outs,
                        config.saguaro_C,
                    )
                    pending_draft_tokens = fb_tokens
                    pending_draft_probs = fb_probs

            round_tokens = list(draft_tokens[:k]) + [bonus_token]
            generated.extend(round_tokens)

            result = RoundResult(
                tokens_accepted=k,
                bonus_token=bonus_token,
                all_tokens=round_tokens,
                wall_time_s=wall_time,
                cache_hit=cache_hit,
                draft_probs=acc_dp,
                target_probs=acc_tp,
            )
            accumulator.record(result)

            if bonus_token == tokenizer.eos_token_id:
                break

    text = decode(generated, tokenizer)
    metrics = accumulator.finalize(K=config.K)
    return text, metrics
