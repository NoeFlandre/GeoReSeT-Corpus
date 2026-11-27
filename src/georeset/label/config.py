"""The benchmark's serving configuration: the pinned model and draft, greedy sampling, engine settings."""

MODEL_ID = "LiquidAI/LFM2.5-2.6B"
MODEL_REVISION = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
DRAFT_ID = "LiquidAI/LFM2.5-2.6B-DSpark"
DRAFT_REVISION = "458cedab07d0f7b2b05700c77e1aa463d43d6f04"
MAX_NEW_TOKENS = 4096
MAX_RUNNING_REQUESTS = 64  # speed only: requests the engine runs at once; it does not change the output

SAMPLING = {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS}
ENGINE = {
    "dtype": "bfloat16",
    "random_seed": 0,
    "speculative_algorithm": "DSPARK",
    "speculative_draft_attention_backend": "flashinfer",
    "disable_radix_cache": True,
    "mem_fraction_static": 0.75,
}


def engine_kwargs() -> dict[str, object]:
    """Arguments of ``sglang.Engine``: the pinned model and draft, then the benchmark's engine settings."""
    return {
        "model_path": MODEL_ID,
        "revision": MODEL_REVISION,
        "speculative_draft_model_path": DRAFT_ID,
        "speculative_draft_model_revision": DRAFT_REVISION,
        "max_running_requests": MAX_RUNNING_REQUESTS,
        **ENGINE,
    }
