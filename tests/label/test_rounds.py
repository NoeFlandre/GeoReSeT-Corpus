import json

import pytest

from georeset.label import config, parse, prompt, rounds


def test_round_name_is_the_model_slug_and_the_prompt_hash():
    assert rounds.name_for("lfm25-dspark", prompt.PROMPT_SHA256) == "lfm25-dspark-p2fb4856"


def test_current_round_pins_the_benchmark_model_prompt_and_parser():
    current = rounds.current("17d7180")
    assert current.name == "lfm25-dspark-p2fb4856"
    assert current.config["model"] == f"{config.MODEL_ID}@{config.MODEL_REVISION}"
    assert current.config["draft"] == f"{config.DRAFT_ID}@{config.DRAFT_REVISION}"
    assert current.config["prompt_sha256"] == prompt.PROMPT_SHA256
    assert current.config["sampling"] == config.SAMPLING
    assert current.config["chat_template_kwargs"] == {"enable_thinking": False}
    assert current.parser_version == parse.PARSER_VERSION
    assert current.code_commits == ["17d7180"]


def test_registry_round_trips_through_json(tmp_path):
    current = rounds.current("17d7180")
    path = tmp_path / "rounds.json"
    rounds.write_registry(path, {current.name: current})
    assert json.loads(path.read_text())[current.name]["prompt_sha256"] == prompt.PROMPT_SHA256
    assert rounds.read_registry(path) == {current.name: current}


def test_registry_keeps_every_round_it_is_given(tmp_path):
    first = rounds.current("17d7180")
    second = rounds.Round(
        name="other-p0000000",
        slug="other",
        config={"prompt_sha256": "0" * 64},
        prompt_sha256="0" * 64,
        parser_version=parse.PARSER_VERSION,
        code_commits=["abc"],
    )
    path = tmp_path / "rounds.json"
    rounds.write_registry(path, {first.name: first, second.name: second})
    assert set(rounds.read_registry(path)) == {first.name, second.name}


def recorded(**overrides):
    base = {
        "config_fingerprint": ["71dd"],
        "model": ["LiquidAI/LFM2.5-2.6B@654f946"],
        "draft": ["LiquidAI/LFM2.5-2.6B-DSpark@458c"],
        "sglang_version": ["0.5.20"],
        "serving_fingerprint": ["s2", "s1"],
        "gpu": ["NVIDIA L40S", "NVIDIA A40"],
        "code_commit": ["c2", "c1"],
    }
    base.update(overrides)
    return base


def test_backfilled_round_keeps_one_output_configuration_and_lists_what_ran_it():
    found = rounds.backfilled(recorded())
    assert found.name == "lfm25-dspark-p2fb4856"
    assert found.config == {
        "config_fingerprint": "71dd",
        "model": "LiquidAI/LFM2.5-2.6B@654f946",
        "draft": "LiquidAI/LFM2.5-2.6B-DSpark@458c",
        "sglang_version": "0.5.20",
        "serving_fingerprint": ["s1", "s2"],
        "gpu": ["NVIDIA A40", "NVIDIA L40S"],
    }
    assert found.code_commits == ["c1", "c2"]


@pytest.mark.parametrize("field", ["config_fingerprint", "model", "draft", "sglang_version"])
def test_backfilled_round_refuses_two_output_configurations(field):
    with pytest.raises(ValueError, match="refusing to pick one"):
        rounds.backfilled(recorded(**{field: ["a", "b"]}))


def test_backfilled_round_refuses_a_missing_output_configuration():
    fields = recorded()
    del fields["model"]
    with pytest.raises(ValueError, match="0 values for model"):
        rounds.backfilled(fields)


def test_add_keeps_the_registered_rounds_and_adds_the_new_one(tmp_path):
    first = rounds.current("17d7180")
    second = rounds.Round(
        name="other-p0000000",
        slug="other",
        config={"prompt_sha256": "0" * 64},
        prompt_sha256="0" * 64,
        parser_version=parse.PARSER_VERSION,
        code_commits=["abc"],
    )
    path = tmp_path / "rounds.json"
    rounds.write_registry(path, {first.name: first})
    assert set(rounds.add(path, second)) == {first.name, second.name}


def test_add_starts_the_registry_when_there_is_none(tmp_path):
    current = rounds.current("17d7180")
    assert rounds.add(tmp_path / "rounds.json", current) == {current.name: current}
