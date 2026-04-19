"""Shared utility functions for model loading, inference, and token operations."""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_models(
    draft_name: str,
    target_name: str,
    device: str,
) -> tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
    """Load draft and target models in eval mode with a shared tokenizer.

    Args:
        draft_name: HuggingFace model identifier for the draft model.
        target_name: HuggingFace model identifier for the target model.
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        Tuple of (draft_model, target_model, tokenizer) where the tokenizer
        is taken from the target model.
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(target_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        draft_name,
        torch_dtype=torch.float16,
    ).to(device).eval()

    target_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        target_name,
        torch_dtype=torch.float16,
    ).to(device).eval()

    return draft_model, target_model, tokenizer


def get_logits(
    model: PreTrainedModel,
    input_ids: Tensor,
    past_key_values: object | None,
) -> tuple[Tensor, object]:
    """Run one forward pass and return last-position logits plus updated KV cache.

    Args:
        model: A causal language model.
        input_ids: Token id tensor of shape ``(1, seq_len)``.
        past_key_values: Existing KV cache or ``None`` for the first call.

    Returns:
        Tuple of (logits, past_key_values) where logits has shape ``(vocab_size,)``.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
    last_logits: Tensor = outputs.logits[0, -1, :]
    return last_logits, outputs.past_key_values


def logits_to_probs(logits: Tensor, temperature: float) -> Tensor:
    """Convert raw logits to a probability distribution using temperature scaling.

    Args:
        logits: Raw logit tensor of shape ``(vocab_size,)``.
        temperature: Sampling temperature. Values < 1 sharpen; values > 1 flatten.

    Returns:
        Probability tensor of shape ``(vocab_size,)`` summing to 1.
    """
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs
    scaled = logits / temperature
    return torch.softmax(scaled, dim=-1)


def top_k_token_ids(logits: Tensor, k: int) -> list[int]:
    """Return the top-k token ids ranked by descending logit value.

    Args:
        logits: Raw logit tensor of shape ``(vocab_size,)``.
        k: Number of top tokens to return.

    Returns:
        List of ``k`` token ids sorted by descending logit value.
    """
    topk = torch.topk(logits, k=min(k, logits.size(-1)))
    return topk.indices.tolist()


def sample_token(logits: Tensor, temperature: float) -> int:
    """Sample a single token id from the distribution implied by logits.

    Args:
        logits: Raw logit tensor of shape ``(vocab_size,)``.
        temperature: Sampling temperature passed to :func:`logits_to_probs`.

    Returns:
        A single integer token id.
    """
    probs = logits_to_probs(logits, temperature)
    return torch.multinomial(probs, num_samples=1).item()


def encode(text: str, tokenizer: PreTrainedTokenizer, device: str) -> Tensor:
    """Tokenize a text string and move the result to the specified device.

    Args:
        text: Input text to tokenize.
        tokenizer: HuggingFace tokenizer instance.
        device: Torch device string, e.g. ``"cuda"``.

    Returns:
        Integer tensor of shape ``(1, seq_len)`` on ``device``.
    """
    encoded = tokenizer(text, return_tensors="pt")
    return encoded["input_ids"].to(device)


def decode(token_ids: list[int], tokenizer: PreTrainedTokenizer) -> str:
    """Decode a list of token ids back to a human-readable string.

    Args:
        token_ids: List of integer token ids.
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        Decoded text string.
    """
    return tokenizer.decode(token_ids, skip_special_tokens=True)
