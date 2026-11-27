"""The benchmark prompt, verbatim: the digest pins it and ``render`` inserts one sentence."""

from importlib import resources

PLACEHOLDER = "{}"
PROMPT_SHA256 = "2fb48569c8bbb73fcf584fb7549b34ddd5e4cc365b7a5c75be7429906e312097"


class PromptError(ValueError):
    """The prompt is not the benchmark's, or has no slot for the sentence."""


def text() -> str:
    """The vendored prompt file, byte for byte."""
    return (resources.files(__package__) / "prompt.txt").read_bytes().decode("utf-8")


def check_digest(digest: str) -> None:
    """Refuse any prompt other than the one the benchmark scored."""
    if digest != PROMPT_SHA256:
        raise PromptError(f"prompt sha256 {digest} != benchmark {PROMPT_SHA256}")


def render(template: str, sentence: str) -> str:
    """Insert ``sentence`` at the first ``{}``; braces inside the sentence stay as they are."""
    if PLACEHOLDER not in template:
        raise PromptError(f"prompt template has no {PLACEHOLDER!r} placeholder")
    return template.replace(PLACEHOLDER, sentence, 1)
