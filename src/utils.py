"""Shared utility functions for model loading, inference, and token operations."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_models(
    draft_name: str,
    target_name: str,
    device: Optional[str] = None,
    *,
    draft_device: str = "cuda:0",
    target_device: str = "cuda:1",
    dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
) -> tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]:
    """Load draft and target models in eval mode with a shared tokenizer.

    Backward-compatible positional ``device`` argument: when given (e.g.
    ``"cuda"`` or ``"cpu"``) both draft and target are placed on that same
    device, matching the original single-device behaviour. Otherwise the
    keyword-only ``draft_device`` and ``target_device`` are used.

    Tokenizer compatibility is asserted by comparing the vocab of the draft
    and target tokenizers. A mismatch raises ``ValueError`` because
    speculative verification across different vocabularies silently corrupts
    output.

    Args:
        draft_name: HuggingFace model identifier for the draft model.
        target_name: HuggingFace model identifier for the target model.
        device: Optional legacy single-device string. When provided it
            overrides both ``draft_device`` and ``target_device``.
        draft_device: Torch device string for the draft model.
        target_device: Torch device string for the target model.
        dtype: Torch dtype passed via ``torch_dtype`` to ``from_pretrained``.
        device_map: Optional HuggingFace ``device_map`` value. When given,
            the models are loaded with that map and ``.to(device)`` is not
            called.
        trust_remote_code: Forwarded to ``from_pretrained``.

    Returns:
        Tuple of (draft_model, target_model, tokenizer) where the tokenizer
        is taken from ``target_name``.

    Raises:
        ValueError: If the draft and target tokenizers do not share the
            same vocabulary.
    """
    if device is not None:
        draft_device = device
        target_device = device

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        target_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft_tokenizer = AutoTokenizer.from_pretrained(
        draft_name, trust_remote_code=trust_remote_code
    )
    if draft_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError(
            "Draft and target tokenizers have different vocabularies. "
            f"Draft={draft_name!r}, target={target_name!r}. Speculative "
            "verification requires matching vocab; pick models from the "
            "same tokenizer family."
        )

    draft_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    target_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if device_map is not None:
        draft_kwargs["device_map"] = device_map
        target_kwargs["device_map"] = device_map

    draft_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        draft_name, **draft_kwargs
    )
    target_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        target_name, **target_kwargs
    )

    if device_map is None:
        draft_model = draft_model.to(draft_device)
        target_model = target_model.to(target_device)

    draft_model.eval()
    target_model.eval()

    return draft_model, target_model, tokenizer


def sync_if_cuda(device: torch.device | str | None) -> None:
    """Call ``torch.cuda.synchronize`` on ``device`` if it is a CUDA device.

    No-op for CPU devices, so timing code remains valid in CPU smoke tests.
    Required around every ``time.perf_counter()`` pair that wraps GPU work,
    because CUDA launches are asynchronous and unsynced wall-clocks measure
    only kernel-launch overhead.

    Args:
        device: A ``torch.device``, device string (``"cuda:0"``, ``"cpu"``),
            or ``None``. When ``None``, synchronises the current default
            CUDA device if CUDA is available; otherwise no-op.

    Returns:
        None.
    """
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def is_eos(token_id: int, tokenizer: PreTrainedTokenizerBase) -> bool:
    """Return True if ``token_id`` matches any of the tokenizer's EOS ids.

    Handles three cases: scalar int ``eos_token_id`` (GPT-2 style),
    list/tuple ``eos_token_id`` (Qwen/Llama style), or ``None`` (no EOS
    defined).

    Args:
        token_id: Candidate token id to test.
        tokenizer: HuggingFace tokenizer whose ``eos_token_id`` is consulted.

    Returns:
        ``True`` when ``token_id`` matches any configured EOS id.
    """
    eos = tokenizer.eos_token_id
    if eos is None:
        return False
    if isinstance(eos, (list, tuple)):
        return int(token_id) in eos
    return int(token_id) == int(eos)


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


def encode(text: str, tokenizer: PreTrainedTokenizerBase, device: str) -> Tensor:
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


def decode(token_ids: list[int], tokenizer: PreTrainedTokenizerBase) -> str:
    """Decode a list of token ids back to a human-readable string.

    Args:
        token_ids: List of integer token ids.
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        Decoded text string.
    """
    return tokenizer.decode(token_ids, skip_special_tokens=True)
