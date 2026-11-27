from georeset.label import config


def test_model_and_draft_are_pinned_to_the_benchmark_revisions():
    kwargs = config.engine_kwargs()
    assert kwargs["model_path"] == "LiquidAI/LFM2.5-2.6B"
    assert kwargs["revision"] == "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
    assert kwargs["speculative_draft_model_path"] == "LiquidAI/LFM2.5-2.6B-DSpark"
    assert kwargs["speculative_draft_model_revision"] == "458cedab07d0f7b2b05700c77e1aa463d43d6f04"


def test_engine_is_the_benchmark_engine_with_the_speed_settings_added():
    kwargs = config.engine_kwargs()
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["random_seed"] == 0
    assert kwargs["speculative_algorithm"] == "DSPARK"
    assert kwargs["mem_fraction_static"] == 0.75
    assert kwargs["max_running_requests"] == config.MAX_RUNNING_REQUESTS


def test_sampling_is_greedy_with_the_benchmark_budget():
    assert config.SAMPLING == {"temperature": 0.0, "max_new_tokens": 4096}
