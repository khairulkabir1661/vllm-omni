# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for diffusion benchmark tokenizer bypass (Issue #6873)."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from vllm_omni.benchmarks.serve import (
    _DIFFUSION_BACKENDS,
    _DIFFUSION_ENDPOINTS,
    _prepare_diffusion_args,
    is_diffusion_benchmark,
    main,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cpu,
]


@pytest.mark.parametrize("endpoint", sorted(_DIFFUSION_ENDPOINTS))
def test_diffusion_endpoints_detected(endpoint):
    assert is_diffusion_benchmark(Namespace(endpoint=endpoint, backend=""))


@pytest.mark.parametrize("endpoint", sorted(_DIFFUSION_ENDPOINTS))
def test_diffusion_endpoints_detected_without_leading_slash(endpoint):
    assert is_diffusion_benchmark(Namespace(endpoint=endpoint.lstrip("/"), backend=""))


@pytest.mark.parametrize("backend", sorted(_DIFFUSION_BACKENDS))
def test_diffusion_backends_detected(backend):
    assert is_diffusion_benchmark(Namespace(endpoint="", backend=backend))


@pytest.mark.parametrize(
    "endpoint,backend",
    [
        ("/v1/chat/completions", ""),
        ("", "openai-chat-omni"),
        ("", ""),
    ],
)
def test_non_diffusion_not_detected(endpoint, backend):
    assert not is_diffusion_benchmark(Namespace(endpoint=endpoint, backend=backend))


def test_prepare_diffusion_args_sets_skip_tokenizer():
    args = Namespace()
    _prepare_diffusion_args(args)
    assert args.skip_tokenizer_init is True


def test_prepare_diffusion_args_respects_explicit_tokenizer():
    args = Namespace(tokenizer="gpt2")
    _prepare_diffusion_args(args)
    assert not getattr(args, "skip_tokenizer_init", False)


def _make_bench_args(**overrides):
    defaults = dict(
        endpoint="/v1/images/generations",
        backend="",
        seed_tts_wer_eval=False,
        seed_tts_wer_save_items=False,
        daily_omni_save_eval_items=False,
        videomme_save_eval_items=False,
        omni_request_timeout_s=None,
        print_stage=False,
        extra_body=None,
        dataset_name=None,
        tokenizer=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_main_sets_skip_tokenizer_for_diffusion():
    args = _make_bench_args()
    with patch(
        "vllm_omni.benchmarks.serve.main_async",
        new_callable=AsyncMock,
        return_value={"ok": True},
    ) as mock:
        result = main(args)
        assert mock.called
        assert result == {"ok": True}
        called_args = mock.call_args[0][0]
        assert called_args.skip_tokenizer_init is True


def test_main_does_not_skip_tokenizer_when_explicit():
    args = _make_bench_args(tokenizer="gpt2")
    with patch(
        "vllm_omni.benchmarks.serve.main_async",
        new_callable=AsyncMock,
        return_value={"ok": True},
    ) as mock:
        main(args)
        called_args = mock.call_args[0][0]
        assert not getattr(called_args, "skip_tokenizer_init", False)


def test_main_slashless_endpoint_sets_skip_tokenizer():
    args = _make_bench_args(endpoint="v1/videos")
    with patch(
        "vllm_omni.benchmarks.serve.main_async",
        new_callable=AsyncMock,
        return_value={"ok": True},
    ) as mock:
        main(args)
        called_args = mock.call_args[0][0]
        assert called_args.skip_tokenizer_init is True
