import pytest

from georeset.label.text import has_no_letters, sha256_text


@pytest.mark.parametrize("text", ["", "  ", "12 345", "?!...", "🙂 ...", "1990-2000"])
def test_texts_without_letters_are_letterless(text):
    assert has_no_letters(text)


@pytest.mark.parametrize("text", ["a", "Été", "漢字", "Привет", "مرحبا", "1 km²"])
def test_any_unicode_letter_counts(text):
    assert not has_no_letters(text)


def test_sha256_is_of_the_exact_utf8_bytes():
    assert sha256_text("a") == "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
    assert sha256_text("é") != sha256_text("é")  # no normalisation
