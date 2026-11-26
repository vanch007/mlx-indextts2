"""Tests for MLX memory/cache limit resolution and per-utterance cache release."""

import numpy as np
import pytest

from mlx_indextts.performance import BYTES_PER_GB, resolve_mlx_memory_limits


@pytest.mark.parametrize(
    ("total_gb", "expected_mem", "expected_cache"),
    [
        (8, 4.0, 0.5),
        (16, 8.0, 1.0),
        (24, 12.0, 1.5),
        (32, 16.0, 2.0),
        (48, 24.0, 3.0),
        (64, 48.0, 16.0),
        (128, 96.0, 32.0),
    ],
)
def test_resolve_mlx_memory_limits_scales_with_unified_memory(
    monkeypatch, total_gb, expected_mem, expected_cache
):
    for var in (
        "MLX_INDEXTTS_MEMORY_LIMIT_GB",
        "MLX_TTS_MEMORY_LIMIT_GB",
        "MLX_MEMORY_LIMIT_GB",
        "MLX_INDEXTTS_CACHE_LIMIT_GB",
        "MLX_TTS_CACHE_LIMIT_GB",
        "MLX_CACHE_LIMIT_GB",
    ):
        monkeypatch.delenv(var, raising=False)

    limits = resolve_mlx_memory_limits(total_memory_bytes=int(total_gb * BYTES_PER_GB))
    assert limits.memory_limit_gb == expected_mem
    assert limits.cache_limit_gb == expected_cache
    assert limits.source == "auto"


def test_runtime_generate_clears_mlx_cache_after_each_utterance(monkeypatch, tmp_path):
    import mlx_indextts.runtime as runtime_module

    cleared = []

    class FakeTTS:
        model_revision = "rev"
        last_generation_info = {"resolved_language": "zh"}
        use_gpt_latent = False

        def generate(self, **kwargs):
            return np.zeros(2205, dtype=np.float32)

    monkeypatch.setattr(runtime_module, "detect_mlx_version", lambda _p: "2.5")
    monkeypatch.setattr(runtime_module, "_clear_mlx_cache", lambda: cleared.append(True))
    runtime = runtime_module.TTSRuntime(quantize="fp32")
    monkeypatch.setattr(runtime, "load", lambda _p: FakeTTS())
    monkeypatch.setattr(
        runtime,
        "_resolve_auto_emotion",
        lambda _t, _o: (None, {"emotion_source": "speaker_reference"}),
    )

    for idx in range(2):
        runtime.generate(
            text="测试",
            ref_audio="speaker.npz",
            output_path=str(tmp_path / f"out_{idx}.wav"),
            model="model-v25",
            options=runtime_module.GenerateOptions(
                denoise_ref_audio=False,
                denoise_emotion_ref_audio=False,
            ),
        )

    assert len(cleared) == 2


def test_runtime_stream_clears_mlx_cache_on_completion(monkeypatch):
    import mlx_indextts.runtime as runtime_module

    cleared = []

    class FakeTTS:
        last_generation_info = {}
        use_gpt_latent = False

        def stream(self, **kwargs):
            yield np.zeros(1024, dtype=np.float32)

    monkeypatch.setattr(runtime_module, "detect_mlx_version", lambda _p: "2.5")
    monkeypatch.setattr(runtime_module, "_clear_mlx_cache", lambda: cleared.append(True))
    runtime = runtime_module.TTSRuntime(quantize="fp32")
    monkeypatch.setattr(runtime, "load", lambda _p: FakeTTS())
    monkeypatch.setattr(
        runtime,
        "_resolve_auto_emotion",
        lambda _t, _o: (None, {"emotion_source": "speaker_reference"}),
    )

    chunks = list(
        runtime.stream(
            text="测试",
            ref_audio="speaker.npz",
            model="model-v25",
            options=runtime_module.GenerateOptions(
                denoise_ref_audio=False,
                denoise_emotion_ref_audio=False,
            ),
        )
    )

    assert len(chunks) == 1
    assert cleared == [True]
