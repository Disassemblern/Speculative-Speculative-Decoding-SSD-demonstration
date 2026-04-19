"""Standard autoregressive text generation with KV-cache acceleration."""

from __future__ import annotations

import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.metrics import MetricsAccumulator, RoundResult, SessionMetrics
from src.utils import decode, encode, get_logits, sample_token


def generate_autoregressive(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> tuple[str, SessionMetrics]:
    """Generate text token-by-token using a standard autoregressive loop.

    Each generated token is treated as one round with ``tokens_accepted=0``
    and the produced token as the bonus token. KV cache is reused across
    tokens for efficiency.

    Args:
        model: A causal language model in eval mode.
        tokenizer: Tokenizer compatible with ``model``.
        prompt: Input text to condition generation on.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        device: Torch device string, e.g. ``"cuda"``.

    Returns:
        Tuple of (generated_text, session_metrics).
    """
    input_ids = encode(prompt, tokenizer, device)
    accumulator = MetricsAccumulator()
    generated: list[int] = []
    past_key_values = None
    current_ids = input_ids

    for _ in range(max_new_tokens):
        t_start = time.perf_counter()

        logits, past_key_values = get_logits(model, current_ids, past_key_values)
        token_id: int = sample_token(logits, temperature)

        wall_time = time.perf_counter() - t_start

        result = RoundResult(
            tokens_accepted=0,
            bonus_token=token_id,
            all_tokens=[token_id],
            wall_time_s=wall_time,
            cache_hit=None,
            draft_probs=[],
            target_probs=[],
        )
        accumulator.record(result)
        generated.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

        current_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)

    text = decode(generated, tokenizer)
    metrics = accumulator.finalize(K=0)
    return text, metrics
