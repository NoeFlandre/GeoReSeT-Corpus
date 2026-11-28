"""Labelling rounds: each round is one sidecar file, and the registry records every round's configuration.

A round is named from its model slug and its prompt hash, e.g. ``lfm25-dspark-p2fb4856``. The registry
(``labels/rounds.json`` in the dataset) holds each round's configuration, so a round can be read without
the code that produced it.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from georeset.label import config, parse, prompt, tokenizer

SLUG = "lfm25-dspark"  # the model and draft of the current configuration
# What decides the output: a round exists only if every generation agrees on it.
SINGLE = ("config_fingerprint", "model", "draft", "sglang_version")
# What ran it (hardware, speed settings, code versions): many values are expected, so they are listed.
MANY = ("serving_fingerprint", "gpu")


@dataclass(frozen=True, slots=True)
class Round:
    name: str
    slug: str
    config: dict[str, Any]
    prompt_sha256: str
    parser_version: str
    code_commits: list[str]


def name_for(slug: str, prompt_sha256: str) -> str:
    return f"{slug}-p{prompt_sha256[:7]}"


def current(code_commit: str) -> Round:
    """The round the labelling code produces: the benchmark's configuration, from this commit."""
    return Round(
        name=name_for(SLUG, prompt.PROMPT_SHA256),
        slug=SLUG,
        config={
            "model": f"{config.MODEL_ID}@{config.MODEL_REVISION}",
            "draft": f"{config.DRAFT_ID}@{config.DRAFT_REVISION}",
            "prompt_sha256": prompt.PROMPT_SHA256,
            "chat_template_kwargs": tokenizer.CHAT_TEMPLATE_KWARGS,
            "sampling": config.SAMPLING,
            "engine": {**config.ENGINE, "max_running_requests": config.MAX_RUNNING_REQUESTS},
        },
        prompt_sha256=prompt.PROMPT_SHA256,
        parser_version=parse.PARSER_VERSION,
        code_commits=[code_commit],
    )


def backfilled(recorded: dict[str, list[str]]) -> Round:
    """Round 1 from the generations' provenance: one output configuration, and everything that ran it.

    The labels were produced with the benchmark's prompt, so the name uses its hash.
    """
    for field in SINGLE:
        values = recorded.get(field, [])
        if len(values) != 1:
            raise ValueError(f"the generations record {len(values)} values for {field}; refusing to pick one")
    fields: dict[str, Any] = {field: recorded[field][0] for field in SINGLE}
    fields |= {field: sorted(recorded.get(field, [])) for field in MANY}
    return Round(
        name=name_for(SLUG, prompt.PROMPT_SHA256),
        slug=SLUG,
        config=fields,
        prompt_sha256=prompt.PROMPT_SHA256,
        parser_version=parse.PARSER_VERSION,
        code_commits=sorted(recorded.get("code_commit", [])),
    )


def write_registry(path: Path, rounds: dict[str, Round]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {name: asdict(found) for name, found in sorted(rounds.items())}
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_registry(path: Path) -> dict[str, Round]:
    return {name: Round(**fields) for name, fields in json.loads(path.read_text(encoding="utf-8")).items()}


def add(path: Path, found: Round) -> dict[str, Round]:
    """The registry at ``path`` (none yet: empty) with ``found`` added; the rounds already there stay."""
    existing = read_registry(path) if path.exists() else {}
    return {**existing, found.name: found}
