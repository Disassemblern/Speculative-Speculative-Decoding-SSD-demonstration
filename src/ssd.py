"""Speculative Speculative Decoding (SSD) with Saguaro cache pre-computation.

Implements the paper-faithful variant where Saguaro cache construction runs in
parallel with target verification. The cache build is split into two phases:

* ``build_rejection_positions``: covers rejection outcomes ``k in 0..K-1``.
  These entries depend only on draft-side tensors (``raw_logits_list`` and
  ``draft_tokens``) and so can be filled IN PARALLEL with verification.
* ``build_all_accepted``: covers the all-accepted ``k = K`` outcome. This
  position needs ``bonus_probs`` (a target-side quantity), so it can only
  start after verification completes; it then runs in parallel with the
  small rejection-sample CPU step.

Cache entries also store ``next_raw_logits`` so the cache-hit fast path does
NOT need to re-run K draft forwards just to recover raw logits for the next
round's cache build.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.metrics import MetricsAccumulator, RoundResult, SessionMetrics
from src.speculative import rejection_sample, verify_draft_tokens
from src.utils import decode, encode, is_eos, logits_to_probs, sync_if_cuda, top_k_token_ids


_LOGGER = logging.getLogger(__name__)
_CLONE_WARNED = False


def _clone_kv_cache(past_kv: object) -> object:
    """Deep-copy a HuggingFace KV cache by cloning every key/value tensor.

    Supports both the modern ``DynamicCache`` object (transformers >= 4.36)
    and the legacy tuple-of-tuples format. ``tensor.clone()`` is used (not
    :func:`copy.deepcopy`) because some attention implementations attach
    non-copyable handles to the cache object; cloning the leaf tensors and
    rebuilding the container is safer.

    Args:
        past_kv: A ``DynamicCache``, legacy tuple-of-tuples cache, or ``None``.

    Returns:
        A new cache object with all tensors cloned, or ``None`` when the
        input was ``None``. Falls back to the legacy tuple format when
        ``DynamicCache.from_legacy_cache`` is not importable.
    """
    if past_kv is None:
        return None
    if hasattr(past_kv, "to_legacy_cache"):
        legacy = past_kv.to_legacy_cache()
        cloned_legacy = tuple(
            (k.clone(), v.clone()) for k, v in legacy
        )
        try:
            from transformers.cache_utils import DynamicCache

            return DynamicCache.from_legacy_cache(cloned_legacy)
        except ImportError:
            return cloned_legacy
    return tuple((k.clone(), v.clone()) for k, v in past_kv)


@dataclass
class SSDConfig:
    """Configuration for Speculative Speculative Decoding.

    Attributes:
        K: Number of draft tokens per round.
        temperature: Sampling temperature for both models.
        device: Torch device string.
        cache_budget_B: Total number of cache entries to pre-fill. Defaults
            to 8 because each cache entry costs K sequential draft forwards;
            larger budgets quickly dominate per-round wall time.
        saguaro_C: Concentration factor for top-F token upweighting (0 < C <= 1).
        acceptance_rate_estimate: Prior estimate of draft acceptance rate.
        power_law_r: Power-law exponent for Theorem 12 fan-out weights.
        fallback: Strategy when cache misses; ``"neural"`` re-runs draft model.
    """

    K: int = 4
    temperature: float = 1.0
    device: str = "cuda"
    cache_budget_B: int = 8
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
        next_draft_probs: Pre-computed (modified) draft probability tensors
            for the next round, used by rejection sampling.
        next_raw_logits: Pre-computed raw (pre-temperature, pre-Saguaro-shift)
            logit tensors at each of the K draft positions. Stored so the
            cache-hit fast path can build the NEXT round's cache without
            re-running K draft forwards just to recover these logits.
    """

    k: int
    bonus_token: int
    next_draft_tokens: list[int]
    next_draft_probs: list[Tensor]
    next_raw_logits: list[Tensor]


class SaguaroCache:
    """Pre-computes and stores draft tokens for all likely next contexts.

    Uses a geometric fan-out strategy (Theorem 12) to allocate the cache
    budget across the K rejection positions plus the all-accepted case.
    The cache build is split into two phases (rejection positions vs.
    all-accepted) so phase 1 can run in parallel with target verification.
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

    def _clear_and_prepare(self) -> None:
        """Reset the internal store before a fresh round of cache filling.

        Returns:
            None.
        """
        self._store.clear()

    def build_rejection_positions(
        self,
        draft_model: PreTrainedModel,
        context_ids: Tensor,
        draft_tokens: list[int],
        raw_logits_list: list[Tensor],
        temperature: float,
    ) -> None:
        """Pre-fill cache entries for rejection positions ``k`` in ``0..K-1``.

        These entries depend only on ``raw_logits_list`` (draft side) and the
        already-sampled ``draft_tokens``. They can therefore be computed
        BEFORE target verification finishes, which is the whole point of
        Saguaro cache overlap.

        Optimisation: instead of running ``K`` full-context draft forwards
        per candidate, the shared prefix ``context + draft_tokens[:k]`` is
        primed into a single KV cache, cloned per candidate, and then each
        candidate only needs ``K`` single-token forwards. Across positions
        ``k = 0..K-1`` the prime is extended one token at a time, costing
        only ``K - 1`` extra single-token forwards in total.

        This method clears the cache before filling, so it must be the FIRST
        of the two build phases invoked per round.

        Args:
            draft_model: The smaller draft language model.
            context_ids: Current context token ids of shape ``(1, context_len)``.
            draft_tokens: The K draft tokens used in the current round.
            raw_logits_list: Raw logit tensors (pre-temperature) at each of the
                K draft positions, used to select top-F_k candidates.
            temperature: Sampling temperature.

        Returns:
            None.
        """
        self._clear_and_prepare()
        fan_outs = self.compute_fan_out()
        K = len(raw_logits_list)
        try:
            device = next(draft_model.parameters()).device
        except (AttributeError, StopIteration):
            device = context_ids.device
        context_ids = context_ids.to(device)

        # Prime KV cache over the original `context` once. For position k > 0
        # we extend by feeding draft_tokens[k-1] one token at a time.
        prime_kv: object = None
        prime_failed = False
        try:
            with torch.no_grad():
                out = draft_model(
                    input_ids=context_ids,
                    past_key_values=None,
                    use_cache=True,
                )
                prime_kv = out.past_key_values
        except Exception as exc:  # noqa: BLE001
            prime_failed = True
            self._warn_clone_fallback(exc)

        for k in range(K):
            F_k = fan_outs[k] if k < len(fan_outs) else 1
            candidates = top_k_token_ids(raw_logits_list[k], F_k)

            if not prime_failed:
                try:
                    self._fill_entries_for_position_primed(
                        draft_model=draft_model,
                        prime_kv=prime_kv,
                        candidates=candidates,
                        k=k,
                        K=K,
                        temperature=temperature,
                        device=device,
                    )
                except Exception as exc:  # noqa: BLE001
                    prime_failed = True
                    self._warn_clone_fallback(exc)

            if prime_failed:
                prefix = list(draft_tokens[:k])
                self._fill_entries_for_position(
                    draft_model=draft_model,
                    context_ids=context_ids,
                    prefix=prefix,
                    candidates=candidates,
                    k=k,
                    K=K,
                    temperature=temperature,
                    device=device,
                )
                continue

            # Extend prime_kv by feeding draft_tokens[k] for the NEXT k.
            if k < K - 1:
                next_tok = torch.tensor(
                    [[int(draft_tokens[k])]], dtype=torch.long, device=device
                )
                try:
                    with torch.no_grad():
                        out = draft_model(
                            input_ids=next_tok,
                            past_key_values=prime_kv,
                            use_cache=True,
                        )
                        prime_kv = out.past_key_values
                except Exception as exc:  # noqa: BLE001
                    prime_failed = True
                    self._warn_clone_fallback(exc)

    def build_all_accepted(
        self,
        draft_model: PreTrainedModel,
        context_ids: Tensor,
        draft_tokens: list[int],
        bonus_probs: Tensor,
        temperature: float,
    ) -> None:
        """Pre-fill cache entries for the all-accepted position ``k = K``.

        These entries select candidate bonus tokens from ``bonus_probs``, a
        target-side distribution. This phase therefore must run AFTER target
        verification completes. It is typically tiny (fan_out at position K
        is hard-coded to 1) and overlaps with the rejection-sample CPU step.

        Args:
            draft_model: The smaller draft language model.
            context_ids: Current context token ids of shape ``(1, context_len)``.
            draft_tokens: The K draft tokens used in the current round.
            bonus_probs: Target probability tensor for the all-accepted position.
            temperature: Sampling temperature.

        Returns:
            None.
        """
        fan_outs = self.compute_fan_out()
        K = len(draft_tokens)
        try:
            device = next(draft_model.parameters()).device
        except (AttributeError, StopIteration):
            device = context_ids.device
        context_ids = context_ids.to(device)

        F_K = fan_outs[K] if K < len(fan_outs) else 1
        candidates = top_k_token_ids(bonus_probs, F_K)

        # Prime KV cache over `context + draft_tokens` once, then each
        # candidate only costs K single-token forwards. Falls back to the
        # original full-context path if priming/cloning fails for any
        # reason (e.g., unusual cache type in a future transformers).
        prime_failed = False
        prime_kv: object = None
        try:
            prefix_tokens = torch.tensor(
                [list(draft_tokens)], dtype=torch.long, device=device
            )
            prefix_ids = (
                torch.cat([context_ids, prefix_tokens], dim=1)
                if draft_tokens
                else context_ids
            )
            with torch.no_grad():
                out = draft_model(
                    input_ids=prefix_ids,
                    past_key_values=None,
                    use_cache=True,
                )
                prime_kv = out.past_key_values
        except Exception as exc:  # noqa: BLE001
            prime_failed = True
            self._warn_clone_fallback(exc)

        if not prime_failed:
            try:
                self._fill_entries_for_position_primed(
                    draft_model=draft_model,
                    prime_kv=prime_kv,
                    candidates=candidates,
                    k=K,
                    K=K,
                    temperature=temperature,
                    device=device,
                )
                return
            except Exception as exc:  # noqa: BLE001
                self._warn_clone_fallback(exc)

        prefix = list(draft_tokens)
        self._fill_entries_for_position(
            draft_model=draft_model,
            context_ids=context_ids,
            prefix=prefix,
            candidates=candidates,
            k=K,
            K=K,
            temperature=temperature,
            device=device,
        )

    def build(
        self,
        draft_model: PreTrainedModel,
        context_ids: Tensor,
        draft_tokens: list[int],
        raw_logits_list: list[Tensor],
        bonus_probs: Tensor,
        temperature: float,
    ) -> None:
        """Convenience: run both build phases sequentially.

        Useful for tests and call sites that do not care about overlap with
        verification. Production SSD calls ``build_rejection_positions`` and
        ``build_all_accepted`` separately around the verifier future.

        Args:
            draft_model: The smaller draft language model.
            context_ids: Current context token ids of shape ``(1, context_len)``.
            draft_tokens: The K draft tokens used in the current round.
            raw_logits_list: Raw logit tensors at each draft position.
            bonus_probs: Target probability tensor for the all-accepted position.
            temperature: Sampling temperature.

        Returns:
            None.
        """
        self.build_rejection_positions(
            draft_model, context_ids, draft_tokens, raw_logits_list, temperature
        )
        self.build_all_accepted(
            draft_model, context_ids, draft_tokens, bonus_probs, temperature
        )

    @staticmethod
    def _warn_clone_fallback(exc: BaseException) -> None:
        """Emit a one-time warning when the primed-cache fast path fails.

        Args:
            exc: The exception that triggered the fallback.

        Returns:
            None.
        """
        global _CLONE_WARNED
        if not _CLONE_WARNED:
            _CLONE_WARNED = True
            _LOGGER.warning(
                "SaguaroCache primed KV-cache path failed (%s: %s); "
                "falling back to full-context draft forwards for the rest "
                "of this run.",
                type(exc).__name__,
                exc,
            )

    def _fill_entries_for_position_primed(
        self,
        draft_model: PreTrainedModel,
        prime_kv: object,
        candidates: list[int],
        k: int,
        K: int,
        temperature: float,
        device: torch.device,
    ) -> None:
        """Fast path: fill cache entries by cloning a primed KV cache per candidate.

        For each candidate the prime is cloned (so the shared prefix is not
        mutated), then ``K`` single-token forwards extend the clone. This
        avoids re-running the draft model on the shared prefix once per
        candidate.

        Args:
            draft_model: The smaller draft language model.
            prime_kv: KV cache covering the shared prefix
                (``context + draft_tokens[:k]``). Not mutated by this method.
            candidates: Bonus-token candidates at position ``k``.
            k: The rejection position (``0..K``) the entries belong to.
            K: Number of tokens to draft for the next round.
            temperature: Sampling temperature.
            device: Device the draft model lives on.

        Returns:
            None.

        Raises:
            Exception: Propagates any error from the cache-clone or forward
                call. The caller is responsible for falling back to the
                slow path.
        """
        for bonus_candidate in candidates:
            cloned = _clone_kv_cache(prime_kv)
            next_tokens, next_probs, next_raw = _draft_K_tokens_with_prime(
                draft_model=draft_model,
                prime_kv=cloned,
                first_token_id=int(bonus_candidate),
                K=K,
                temperature=temperature,
                device=device,
            )
            self._store[(k, int(bonus_candidate))] = CacheEntry(
                k=k,
                bonus_token=int(bonus_candidate),
                next_draft_tokens=next_tokens,
                next_draft_probs=next_probs,
                next_raw_logits=next_raw,
            )

    def _fill_entries_for_position(
        self,
        draft_model: PreTrainedModel,
        context_ids: Tensor,
        prefix: list[int],
        candidates: list[int],
        k: int,
        K: int,
        temperature: float,
        device: torch.device,
    ) -> None:
        """Run draft forwards for each candidate at position ``k`` and store entries.

        Args:
            draft_model: The smaller draft language model.
            context_ids: Current context token ids on ``device``.
            prefix: Token ids that precede the bonus candidate at position ``k``.
            candidates: Bonus-token candidates at position ``k``.
            k: The rejection position (``0..K``) the entries belong to.
            K: Number of tokens to draft for the next round.
            temperature: Sampling temperature.
            device: Device the draft model lives on.

        Returns:
            None.
        """
        for bonus_candidate in candidates:
            hyp_tokens = prefix + [int(bonus_candidate)]
            hyp_tensor = torch.tensor(
                hyp_tokens, dtype=torch.long, device=device
            ).unsqueeze(0)
            hyp_context = torch.cat([context_ids, hyp_tensor], dim=1)

            next_tokens, next_probs, next_raw = _draft_K_tokens_with_logits(
                draft_model, hyp_context, K, temperature
            )
            self._store[(k, int(bonus_candidate))] = CacheEntry(
                k=k,
                bonus_token=int(bonus_candidate),
                next_draft_tokens=next_tokens,
                next_draft_probs=next_probs,
                next_raw_logits=next_raw,
            )

    def lookup(self, k: int, bonus_token: int) -> Optional[CacheEntry]:
        """Retrieve a cached draft entry for a given ``(k, bonus_token)`` pair.

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


def _draft_K_tokens_with_prime(
    draft_model: PreTrainedModel,
    prime_kv: object,
    first_token_id: int,
    K: int,
    temperature: float,
    device: torch.device,
) -> tuple[list[int], list[Tensor], list[Tensor]]:
    """Draft K tokens starting from a primed KV cache.

    The KV cache ``prime_kv`` must already cover the shared prefix. This
    helper feeds ``first_token_id`` (the bonus candidate) through the draft
    model first, which produces the logits used to sample the FIRST drafted
    token, and then performs ``K - 1`` further single-token forwards. Total
    cost: exactly ``K`` forwards, each on a single new token, irrespective of
    the prefix length.

    The input ``prime_kv`` is mutated in-place by the forwards, so callers
    that want to reuse the prime MUST clone it via :func:`_clone_kv_cache`
    before calling this helper.

    Args:
        draft_model: The smaller draft language model.
        prime_kv: KV cache covering the shared prefix (mutated in place).
        first_token_id: First token to feed (typically the bonus candidate).
        K: Number of tokens to draft.
        temperature: Sampling temperature.
        device: Torch device for input tensor allocation; should match the
            device of the draft model parameters.

    Returns:
        Tuple ``(token_ids, prob_tensors, raw_logits_tensors)`` each of
        length ``K``. ``token_ids[0]`` is sampled FROM the logits produced
        after feeding ``first_token_id``. Raw logit tensors are detached and
        moved to CPU; probability tensors are moved to CPU; raw logits are
        pre-temperature.
    """
    tokens: list[int] = []
    probs: list[Tensor] = []
    raw_logits: list[Tensor] = []

    current_ids = torch.tensor(
        [[int(first_token_id)]], dtype=torch.long, device=device
    )
    past_kv = prime_kv

    with torch.no_grad():
        for _ in range(K):
            outputs = draft_model(
                input_ids=current_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            past_kv = outputs.past_key_values
            raw_logits.append(logits.detach().to("cpu").clone())
            p = logits_to_probs(logits, temperature)
            t: int = torch.multinomial(p, num_samples=1).item()
            tokens.append(t)
            probs.append(p.cpu())
            current_ids = torch.tensor(
                [[t]], dtype=torch.long, device=device
            )

    return tokens, probs, raw_logits


def _draft_K_tokens_no_cache(
    draft_model: PreTrainedModel,
    input_ids: Tensor,
    K: int,
    temperature: float,
) -> tuple[list[int], list[Tensor]]:
    """Draft K tokens autoregressively without a persistent KV cache.

    Kept for backward compatibility; ``_draft_K_tokens_with_logits`` is
    preferred because it also returns the raw logits required for cache
    construction.

    Args:
        draft_model: The smaller draft language model.
        input_ids: Context tensor of shape ``(1, seq_len)``.
        K: Number of tokens to draft.
        temperature: Sampling temperature.

    Returns:
        Tuple of (token_ids, prob_tensors).
    """
    tokens, probs, _ = _draft_K_tokens_with_logits(
        draft_model, input_ids, K, temperature
    )
    return tokens, probs


def _draft_K_tokens_with_logits(
    draft_model: PreTrainedModel,
    input_ids: Tensor,
    K: int,
    temperature: float,
) -> tuple[list[int], list[Tensor], list[Tensor]]:
    """Draft K tokens and also return the raw logits at each step.

    Used during cache construction so the resulting :class:`CacheEntry` can
    populate ``next_raw_logits`` without an additional pass.

    Args:
        draft_model: The smaller draft language model.
        input_ids: Context tensor of shape ``(1, seq_len)``.
        K: Number of tokens to draft.
        temperature: Sampling temperature.

    Returns:
        Tuple of (token_ids, prob_tensors, raw_logit_tensors). Raw logit
        tensors are detached, moved to CPU, and pre-temperature.
    """
    tokens: list[int] = []
    probs: list[Tensor] = []
    raw_logits: list[Tensor] = []
    try:
        model_device = next(draft_model.parameters()).device
    except (AttributeError, StopIteration):
        model_device = input_ids.device
    current_ids = input_ids.to(model_device)
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
            raw_logits.append(logits.detach().to("cpu").clone())
            p = logits_to_probs(logits, temperature)
            t: int = torch.multinomial(p, num_samples=1).item()
            tokens.append(t)
            probs.append(p.cpu())
            current_ids = torch.tensor(
                [[t]], dtype=torch.long, device=model_device
            )

    return tokens, probs, raw_logits


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
    model_device = next(draft_model.parameters()).device
    current_ids = input_ids.to(model_device)
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

            top_indices = torch.topk(
                raw_logits, k=min(F_k, raw_logits.size(0))
            ).indices
            modified_logits[top_indices] += log_c

            probs = logits_to_probs(modified_logits, temperature)
            modified_probs_list.append(probs.cpu())

            token_id: int = torch.multinomial(probs, num_samples=1).item()
            draft_tokens.append(token_id)
            current_ids = torch.tensor(
                [[token_id]], dtype=torch.long, device=model_device
            )

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
    using a ``ThreadPoolExecutor``. The cache build is split into two phases:

    * Phase 1 (rejection positions ``k = 0..K-1``) runs in PARALLEL with
      target verification, since it depends only on draft-side tensors.
    * Phase 2 (all-accepted position ``k = K``) starts AFTER verification
      because it needs ``bonus_probs``; it runs in parallel with the small
      rejection-sample CPU step.

    When the cache has an entry for the realised ``(k, bonus_token)`` outcome,
    the next round's draft, modified probabilities, AND raw logits are taken
    from cache (no extra draft forwards needed). Otherwise the draft model is
    re-run with Saguaro concentration (neural fallback).

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
    pending_raw_logits: Optional[list[Tensor]] = None

    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device

    # Per-phase timing accumulators (seconds). Printed at end of generation.
    timing = {
        "draft_or_hit": 0.0,
        "submit_overhead": 0.0,
        "verify_wait": 0.0,
        "phase1_wait": 0.0,
        "phase2_wait": 0.0,
        "rejection_sample": 0.0,
        "fallback_draft": 0.0,
        "rounds": 0,
        "cache_hits": 0,
    }
    import os
    _ssd_debug = os.environ.get("SSD_TIMING", "0") == "1"

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

            sync_if_cuda(draft_device)
            sync_if_cuda(target_device)
            t_start = time.perf_counter()

            t0 = time.perf_counter()
            if (
                pending_draft_tokens is not None
                and pending_draft_probs is not None
                and pending_raw_logits is not None
                and len(pending_draft_tokens) == K
            ):
                draft_tokens = pending_draft_tokens
                modified_probs_list = pending_draft_probs
                raw_logits_list = pending_raw_logits
                pending_draft_tokens = None
                pending_draft_probs = None
                pending_raw_logits = None
                timing["cache_hits"] += 1
            else:
                draft_tokens, modified_probs_list, raw_logits_list = saguaro_draft_K_tokens(
                    draft_model, context, K, config.temperature, fan_outs, config.saguaro_C
                )
            sync_if_cuda(draft_device)
            timing["draft_or_hit"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            verifier_future = executor.submit(
                verify_draft_tokens,
                target_model,
                context,
                draft_tokens,
                config.temperature,
            )
            phase1_future = executor.submit(
                cache.build_rejection_positions,
                draft_model,
                context,
                draft_tokens,
                raw_logits_list,
                config.temperature,
            )
            timing["submit_overhead"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            target_probs_list, bonus_probs = verifier_future.result()
            timing["verify_wait"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            phase2_future = executor.submit(
                cache.build_all_accepted,
                draft_model,
                context,
                draft_tokens,
                bonus_probs,
                config.temperature,
            )

            k, bonus_token, acc_dp, acc_tp = rejection_sample(
                draft_tokens, modified_probs_list, target_probs_list, bonus_probs
            )
            timing["rejection_sample"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            phase1_future.result()
            timing["phase1_wait"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            phase2_future.result()
            timing["phase2_wait"] += time.perf_counter() - t0

            sync_if_cuda(draft_device)
            sync_if_cuda(target_device)
            wall_time = time.perf_counter() - t_start
            timing["rounds"] += 1

            entry = cache.lookup(k, bonus_token)
            cache_hit: bool

            if entry is not None and len(entry.next_draft_tokens) == K:
                pending_draft_tokens = entry.next_draft_tokens
                pending_draft_probs = entry.next_draft_probs
                pending_raw_logits = entry.next_raw_logits
                cache_hit = True
            else:
                cache_hit = False
                if config.fallback == "neural":
                    t_fb = time.perf_counter()
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
                    fb_tokens, fb_probs, fb_raw = saguaro_draft_K_tokens(
                        draft_model,
                        next_context,
                        K,
                        config.temperature,
                        fan_outs,
                        config.saguaro_C,
                    )
                    pending_draft_tokens = fb_tokens
                    pending_draft_probs = fb_probs
                    pending_raw_logits = [t.detach().to("cpu").clone() for t in fb_raw]
                    sync_if_cuda(draft_device)
                    timing["fallback_draft"] += time.perf_counter() - t_fb

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

            if is_eos(bonus_token, tokenizer):
                break

    text = decode(generated, tokenizer)
    metrics = accumulator.finalize(K=config.K)

    if _ssd_debug and timing["rounds"] > 0:
        r = timing["rounds"]
        print(
            f"[SSD timing] rounds={r} hits={timing['cache_hits']}/{r} "
            f"draft={timing['draft_or_hit']*1000/r:.1f}ms "
            f"verify={timing['verify_wait']*1000/r:.1f}ms "
            f"phase1={timing['phase1_wait']*1000/r:.1f}ms "
            f"phase2={timing['phase2_wait']*1000/r:.1f}ms "
            f"rej={timing['rejection_sample']*1000/r:.1f}ms "
            f"fallback={timing['fallback_draft']*1000/r:.1f}ms "
            f"submit={timing['submit_overhead']*1000/r:.1f}ms",
            flush=True,
        )

    return text, metrics
