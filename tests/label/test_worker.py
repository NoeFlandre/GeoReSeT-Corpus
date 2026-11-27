import asyncio

import pytest

from georeset.label import worker
from georeset.label.engine import ContextLengthExceededError
from georeset.label.records import RULE_NO_LETTERS, from_sglang, input_too_long_failure

from .fakes import FakeEngine, completion


def item(key, text, ids):
    return worker.Item(key, text, list(ids))


def test_letterless_text_is_decided_without_the_model():
    engine = FakeEngine({})
    (result,) = asyncio.run(worker.label(engine, [item("k", "12 ?!", [7])]))
    assert engine.calls == []
    assert (result.finish_reason, result.raw_output) == (RULE_NO_LETTERS, "</think>no")


def test_context_overflow_becomes_a_recorded_failure():
    engine = FakeEngine({(1, 2): ContextLengthExceededError(10, 8)})
    (result,) = asyncio.run(worker.label(engine, [item("k", "Forest.", [1, 2])]))
    assert result == input_too_long_failure("k", 10, 8)
    assert result.raw_output == "luf_failure:input_too_long:10>8"


def test_model_output_is_recorded_as_sglang_reports_it():
    output = completion("</think>yes")
    engine = FakeEngine({(1, 2, 3): output})
    (result,) = asyncio.run(worker.label(engine, [item("k", "Forest.", [1, 2, 3])]))
    assert result == from_sglang("k", output, 3)


def test_results_follow_the_input_order_whatever_finishes_first():
    keys = [(i,) for i in range(5)]
    replies = {key: completion(f"</think>{key[0]}") for key in keys}
    delays = {(0,): 0.05, (1,): 0.03, (2,): 0.0, (3,): 0.01, (4,): 0.02}
    engine = FakeEngine(replies, delays)
    items = [item(f"k{i}", f"Forest {i}.", key) for i, key in enumerate(keys)]
    results = asyncio.run(worker.label(engine, items))
    assert [r.text_sha256 for r in results] == [f"k{i}" for i in range(5)]


@pytest.mark.parametrize("window", [1, 2])
def test_in_flight_requests_never_exceed_the_window(window):
    keys = [(i,) for i in range(6)]
    engine = FakeEngine({key: completion("</think>no") for key in keys}, {key: 0.01 for key in keys})
    items = [item(f"k{i}", f"Forest {i}.", key) for i, key in enumerate(keys)]
    asyncio.run(worker.label(engine, items, window=window))
    assert engine.peak == window
