import pytest

from georeset.label.engine import ContextLengthExceededError, context_length_overflow


def context_error(input_tokens, context_limit):
    return (
        f"The input ({input_tokens} tokens) is longer than the model's context length "
        f"({context_limit} tokens)."
    )


def context_budget_error(input_tokens, max_new_tokens, context_limit, *, total_tokens=None):
    total_tokens = input_tokens + max_new_tokens if total_tokens is None else total_tokens
    return (
        "Requested token count exceeds the model's maximum context length "
        f"of {context_limit} tokens. You requested a total of {total_tokens} "
        f"tokens: {input_tokens} tokens from the input messages and "
        f"{max_new_tokens} tokens for the completion. Please reduce the number "
        "of tokens in the input messages or the completion to fit within the limit."
    )


@pytest.mark.parametrize(
    ("input_tokens", "context_limit", "expected"),
    [
        (330_533, 131_072, (330_533, 131_072, None)),
        (131_073, 131_072, (131_073, 131_072, None)),
        (131_072, 131_072, (131_072, 131_072, None)),
        (131_071, 131_072, None),
    ],
)
def test_context_error_classifier_matches_sglang_context_rejections(input_tokens, context_limit, expected):
    assert context_length_overflow(context_error(input_tokens, context_limit)) == expected


def test_context_error_classifier_ignores_unrelated_or_unstorable_errors():
    assert context_length_overflow("CUDA out of memory") is None
    assert context_length_overflow(context_error(2_147_483_648, 131_072)) is None


def test_context_error_classifier_handles_requested_completion_budget():
    assert context_length_overflow(context_budget_error(127_000, 4_096, 131_072)) == (
        127_000,
        131_072,
        4_096,
    )
    assert context_length_overflow(context_budget_error(126_976, 4_096, 131_072)) is None
    assert (
        context_length_overflow(context_budget_error(127_000, 4_096, 131_072, total_tokens=131_097)) is None
    )


def test_context_length_exceeded_error_keeps_the_counts():
    error = ContextLengthExceededError(10, 8, 4)
    assert (error.input_tokens, error.context_limit, error.max_new_tokens) == (10, 8, 4)
    assert "10 tokens" in str(error)
