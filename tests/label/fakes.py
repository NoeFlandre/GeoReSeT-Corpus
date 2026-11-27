"""Test doubles: an engine that answers from a table, and a bucket on a local folder."""

import asyncio
import shutil
from pathlib import Path


class FakeEngine:
    """Answers each input (keyed by its token ids) after a delay; records the peak in flight."""

    def __init__(self, replies, delays=None):
        self.replies = replies
        self.delays = delays or {}
        self.calls = []
        self.in_flight = 0
        self.peak = 0

    async def generate(self, input_ids):
        key = tuple(input_ids)
        self.calls.append(key)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delays.get(key, 0.0))
            reply = self.replies[key]
            if isinstance(reply, Exception):
                raise reply
            return reply
        finally:
            self.in_flight -= 1


def completion(text, tokens=3):
    return {
        "text": text,
        "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": tokens},
    }


class FakeBucket:
    """The bucket's ``put``/``get`` on a folder: paths are relative to ``root``."""

    def __init__(self, root: Path):
        self.root = root

    def put(self, files):
        for local, path in files:
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(local, target)

    def get(self, files):
        for path, local in files:
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.root / path, local)
