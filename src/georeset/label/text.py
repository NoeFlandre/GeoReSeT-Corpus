"""Text helpers of the labelling: content addressing and the letterless rule."""

import hashlib


def sha256_text(text: str) -> str:
    """Hex sha256 of the exact UTF-8 bytes of ``text`` (no normalisation)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def has_no_letters(text: str) -> bool:
    """True when ``text`` has no alphabetic character at all (such texts are labelled ``no`` by rule).

    Digits, punctuation, symbols, emoji and whitespace do not count; any Unicode letter does.
    """
    return not any(ch.isalpha() for ch in text)
