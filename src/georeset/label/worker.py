"""Label distinct texts through an engine: one request per text, as the benchmark's node runner made them."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from georeset.label.engine import ContextLengthExceededError
from georeset.label.records import Generation, from_sglang, input_too_long_failure, rule_decided_no
from georeset.label.text import has_no_letters

WINDOW = 64  # requests in flight at once (the engine's max_running_requests)


@dataclass(frozen=True, slots=True)
class Item:
    key: str  # the text's sha256
    text: str
    input_ids: list[int]


class AsyncEngine(Protocol):
    async def generate(self, input_ids: list[int]) -> dict: ...


async def label(engine: AsyncEngine, items: Sequence[Item], window: int = WINDOW) -> list[Generation]:
    """One generation per item, in input order, with at most ``window`` requests in flight."""
    gate = asyncio.Semaphore(window)

    async def one(item: Item) -> Generation:
        async with gate:
            return await _one(engine, item)

    return list(await asyncio.gather(*(one(item) for item in items)))


async def _one(engine: AsyncEngine, item: Item) -> Generation:
    if has_no_letters(item.text):
        return rule_decided_no(item.key, len(item.input_ids))
    try:
        output = await engine.generate(item.input_ids)
    except ContextLengthExceededError as exc:
        return input_too_long_failure(item.key, exc.input_tokens, exc.context_limit, exc.max_new_tokens)
    return from_sglang(item.key, output, len(item.input_ids))
