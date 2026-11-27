import hashlib
from importlib import resources

import pytest

from georeset.label import prompt

PROMPT_FILE = resources.files("georeset.label") / "prompt.txt"


def test_vendored_prompt_is_the_benchmark_prompt():
    prompt.check_digest(hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest())


def test_text_is_the_vendored_file():
    assert prompt.text() == PROMPT_FILE.read_text(encoding="utf-8")


def test_other_prompt_is_refused():
    with pytest.raises(prompt.PromptError, match="benchmark"):
        prompt.check_digest("0" * 64)


def test_render_inserts_the_sentence_at_the_placeholder():
    assert prompt.render("S: {} end", "a {} b {x}") == "S: a {} b {x} end"


def test_render_replaces_only_the_first_placeholder():
    assert prompt.render("{} and {}", "x") == "x and {}"


def test_render_requires_the_placeholder():
    with pytest.raises(prompt.PromptError, match=r"no '\{\}' placeholder"):
        prompt.render("no slot", "x")
