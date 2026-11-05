"""Tests for utility helpers in :mod:`src.utils`."""

from __future__ import annotations

from types import SimpleNamespace

from src.utils import is_eos


def _fake_tokenizer(eos_token_id: object) -> SimpleNamespace:
    """Return a minimal tokenizer-like object with a single ``eos_token_id`` field.

    Args:
        eos_token_id: Value to expose as ``tokenizer.eos_token_id``.

    Returns:
        Lightweight object with the requested attribute.
    """
    return SimpleNamespace(eos_token_id=eos_token_id)


def test_is_eos_scalar_int_matches() -> None:
    """Scalar int EOS (GPT-2 style) matches the configured id and rejects others."""
    tok = _fake_tokenizer(50256)
    assert is_eos(50256, tok) is True
    assert is_eos(50255, tok) is False


def test_is_eos_list_matches_any() -> None:
    """List EOS (Qwen2.5 style) matches every configured id and rejects others."""
    tok = _fake_tokenizer([151645, 151643])
    assert is_eos(151645, tok) is True
    assert is_eos(151643, tok) is True
    assert is_eos(50256, tok) is False


def test_is_eos_tuple_matches_any() -> None:
    """Tuple EOS is treated identically to list."""
    tok = _fake_tokenizer((1, 2))
    assert is_eos(1, tok) is True
    assert is_eos(2, tok) is True
    assert is_eos(3, tok) is False


def test_is_eos_none_is_false() -> None:
    """When ``eos_token_id`` is ``None``, no token should be considered EOS."""
    tok = _fake_tokenizer(None)
    assert is_eos(0, tok) is False
    assert is_eos(50256, tok) is False
