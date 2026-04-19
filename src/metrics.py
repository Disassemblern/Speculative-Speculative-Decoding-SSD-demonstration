"""Data classes and accumulator for tracking generation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RoundResult:
    """Outcome of a single speculative decoding round.

    Attributes:
        tokens_accepted: Number of draft tokens accepted by rejection sampling.
        bonus_token: The bonus token id appended after acceptance/rejection.
        all_tokens: Full token list for this round (accepted drafts + bonus).
        wall_time_s: Wall-clock time in seconds for this round.
        cache_hit: Whether the SSD cache supplied the next draft (None for AR/SD).
        draft_probs: Scalar acceptance probabilities for accepted draft positions.
        target_probs: Scalar target probabilities at accepted draft positions.
    """

    tokens_accepted: int
    bonus_token: int
    all_tokens: list[int]
    wall_time_s: float
    cache_hit: Optional[bool]
    draft_probs: list[float]
    target_probs: list[float]


@dataclass
class SessionMetrics:
    """Aggregate statistics for a complete generation session.

    Attributes:
        total_tokens: Total number of tokens generated.
        total_time_s: Total wall-clock time in seconds.
        tokens_per_second: Throughput in tokens per second.
        mean_tokens_per_round: Average tokens produced per decoding round.
        mean_acceptance_rate: Average fraction of draft tokens accepted per round.
        cache_hit_rate: Fraction of rounds with a cache hit (None for AR/SD).
        round_results: List of per-round outcomes.
    """

    total_tokens: int
    total_time_s: float
    tokens_per_second: float
    mean_tokens_per_round: float
    mean_acceptance_rate: float
    cache_hit_rate: Optional[float]
    round_results: list[RoundResult]


@dataclass
class MetricsAccumulator:
    """Collects RoundResult objects and computes aggregate SessionMetrics.

    Attributes:
        _results: Internal list of recorded round results.
    """

    _results: list[RoundResult] = field(default_factory=list)

    def record(self, result: RoundResult) -> None:
        """Append a single round result to the accumulator.

        Args:
            result: The :class:`RoundResult` to store.

        Returns:
            None.
        """
        self._results.append(result)

    def finalize(self, K: int) -> SessionMetrics:
        """Compute and return aggregate metrics over all recorded rounds.

        Args:
            K: The maximum number of draft tokens per round (used for
               acceptance rate normalisation).

        Returns:
            A :class:`SessionMetrics` instance summarising the session.
        """
        results = self._results
        total_tokens = sum(len(r.all_tokens) for r in results)
        total_time = sum(r.wall_time_s for r in results)
        tps = total_tokens / total_time if total_time > 0 else 0.0

        num_rounds = len(results)
        mean_tpr = total_tokens / num_rounds if num_rounds > 0 else 0.0

        acceptance_rates: list[float] = []
        for r in results:
            if K > 0:
                acceptance_rates.append(r.tokens_accepted / K)
            else:
                acceptance_rates.append(0.0)
        mean_ar = sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0.0

        cache_hits = [r.cache_hit for r in results if r.cache_hit is not None]
        cache_hit_rate: Optional[float] = (
            sum(cache_hits) / len(cache_hits) if cache_hits else None
        )

        return SessionMetrics(
            total_tokens=total_tokens,
            total_time_s=total_time,
            tokens_per_second=tps,
            mean_tokens_per_round=mean_tpr,
            mean_acceptance_rate=mean_ar,
            cache_hit_rate=cache_hit_rate,
            round_results=list(results),
        )

    def rolling_cache_hit_rate(self, window: int = 20) -> list[float]:
        """Compute a rolling cache hit rate over the recorded rounds.

        Args:
            window: Number of rounds in each rolling window.

        Returns:
            List of rolling hit rates, one per round from index ``window-1``
            onward. Returns an empty list if no rounds have cache_hit data.
        """
        hits = [r.cache_hit for r in self._results if r.cache_hit is not None]
        if not hits:
            return []
        rolling: list[float] = []
        for i in range(len(hits)):
            start = max(0, i - window + 1)
            window_hits = hits[start : i + 1]
            rolling.append(sum(window_hits) / len(window_hits))
        return rolling

    def rolling_acceptance_rate(self, window: int = 20, K: int = 4) -> list[float]:
        """Compute a rolling acceptance rate over the recorded rounds.

        Args:
            window: Number of rounds in each rolling window.
            K: Maximum draft tokens per round for normalisation.

        Returns:
            List of rolling acceptance rates, one per round.
        """
        if not self._results:
            return []
        rates = [r.tokens_accepted / K if K > 0 else 0.0 for r in self._results]
        rolling: list[float] = []
        for i in range(len(rates)):
            start = max(0, i - window + 1)
            window_rates = rates[start : i + 1]
            rolling.append(sum(window_rates) / len(window_rates))
        return rolling
