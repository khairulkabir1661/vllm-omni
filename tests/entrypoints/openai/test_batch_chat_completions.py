import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_audio import ChatCompletionAudio
from starlette.requests import Request
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse, UsageInfo

from vllm_omni.entrypoints.openai.batch_serving import OmniOpenAIServingChatBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MSG_BATCH: list[list[ChatCompletionMessageParam]] = [
    [ChatCompletionUserMessageParam(role="user", content="What color is the sky?")],
    [ChatCompletionUserMessageParam(role="user", content="What is 2+2?")],
]
collapse = OmniOpenAIServingChatBatch._maybe_collapse_choices


# Helpers for testing creating a __new__ serving instance and ensuring IDs are correct
def _make_raw_request(headers: list[tuple[bytes, bytes]]):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions/batch",
        "headers": headers,
    }
    return Request(scope)


def _make_completion(text):
    return ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
            )
        ],
        usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


def _make_handler(responses):
    handler = OmniOpenAIServingChatBatch.__new__(OmniOpenAIServingChatBatch)
    handler.create_chat_completion = AsyncMock(side_effect=responses)
    return handler


# Helpers for creating choices of different modality types
def _text_choice(text="hello"):
    return ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=text),
    )


def _audio_choice():
    audio = ChatCompletionAudio(id="a1", data="base64audio", expires_at=0, transcript="")
    return ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=None, audio=audio),
    )


def _image_choice():
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
    msg = ChatMessage.model_construct(role="assistant")
    object.__setattr__(msg, "content", content)
    return ChatCompletionResponseChoice.model_construct(
        index=0,
        message=msg,
    )


def _get_subrequest_ids(handler):
    return [call.args[0].request_id for call in handler.create_chat_completion.call_args_list]


@pytest.mark.parametrize("has_header", [True, False])
def test_subrequest_ids_are_unique(has_header: bool):
    """Ensure that a submitted batch request creates unique subrequest IDs"""
    header = None if not has_header else b"user-123"
    headers = [] if not header else [(b"x-request-id", header)]

    handler = _make_handler([_make_completion("hello"), _make_completion("world")])
    request = BatchChatCompletionRequest(
        model="test-model",
        messages=MSG_BATCH,
        seed=42,
    )
    result = asyncio.run(handler.create_batch_chat_completion(request, _make_raw_request(headers)))
    assert not isinstance(result, ErrorResponse)
    assert result.id.startswith("chatcmpl-batch")

    # Ensure request IDs are unique and that if we had a header, it's in all reqs
    req_ids = _get_subrequest_ids(handler)
    assert len(set(req_ids)) == len(req_ids) == 2

    if header is not None:
        assert all(header.decode("utf-8") in req_id for req_id in req_ids)


### Tests for ensuring that the (currently batch specific) choice collapsing is correct
def test_single_text_passthrough():
    result = collapse([_text_choice()])
    assert result.message.content == "hello"


def test_single_image_passthrough():
    result = collapse([_image_choice()])
    assert isinstance(result.message.content, list)
    assert result.message.content[0]["type"] == "image_url"


def test_text_plus_audio():
    result = collapse([_text_choice(), _audio_choice()])
    assert result.message.content == "hello"
    assert result.message.audio.data == "base64audio"


def test_audio_plus_text_order_independent():
    result = collapse([_audio_choice(), _text_choice()])
    assert result.message.content == "hello"
    assert result.message.audio.data == "base64audio"


def test_image_plus_audio():
    result = collapse([_image_choice(), _audio_choice()])
    assert isinstance(result.message.content, list)
    assert result.message.content[0]["type"] == "image_url"
    assert result.message.audio.data == "base64audio"


def test_two_content_choices_raises():
    with pytest.raises(ValueError, match="Multiple content choices cannot be set"):
        collapse([_text_choice(), _text_choice()])


def test_three_choices_raises():
    with pytest.raises(ValueError, match="got 3"):
        collapse([_text_choice(), _audio_choice(), _text_choice()])


# ---------------------------------------------------------------------------
# Tests for render_batch_chat_request (PR B: validate-once)
# ---------------------------------------------------------------------------


def _make_render_handler():
    """Build an OmniOpenAIServingChatBatch with mocked internals for render tests."""
    handler = OmniOpenAIServingChatBatch.__new__(OmniOpenAIServingChatBatch)
    handler._check_model = AsyncMock(return_value=None)
    handler.engine_client = MagicMock()
    handler.engine_client.errored = False
    handler.engine_client.output_modalities = ["text"]
    handler._maybe_get_adapters = MagicMock(return_value=None)
    handler.models = MagicMock()
    handler.models.model_name.return_value = "test-model"
    handler.renderer = MagicMock()
    handler.renderer.get_tokenizer.return_value = MagicMock()
    handler.parser_cls = None
    handler.use_harmony = False
    handler.online_renderer = MagicMock()
    handler.online_renderer.validate_chat_template.return_value = None
    handler.enable_auto_tools = False
    handler.exclude_tools_when_tool_choice_none = False
    handler.chat_template = None
    handler.chat_template_content_format = "string"
    handler.trust_request_chat_template = False
    handler._preprocess_chat = AsyncMock(
        return_value=(
            [{"role": "user", "content": "hi"}],
            [{"prompt_token_ids": [1, 2, 3]}],
        )
    )
    handler._effective_chat_template_kwargs = MagicMock(return_value={})
    handler.create_error_response = MagicMock(
        side_effect=lambda msg, **kw: ErrorResponse(
            error=ErrorInfo(message=msg, type="BadRequestError", code=400),
        )
    )
    handler._resolve_audio_format = MagicMock(return_value="wav")
    return handler


def _make_batch_request(n: int = 2) -> BatchChatCompletionRequest:
    messages = [[ChatCompletionUserMessageParam(role="user", content=f"Question {i}")] for i in range(n)]
    return BatchChatCompletionRequest(model="test-model", messages=messages)


def test_render_model_check_called_once():
    handler = _make_render_handler()
    asyncio.run(handler.render_batch_chat_request(_make_batch_request(5)))
    handler._check_model.assert_called_once()


def test_render_model_check_error_propagates():
    handler = _make_render_handler()
    handler._check_model.return_value = ErrorResponse(
        error=ErrorInfo(message="not found", type="NotFoundError", code=404),
    )
    result = asyncio.run(handler.render_batch_chat_request(_make_batch_request(5)))
    assert isinstance(result, ErrorResponse)
    handler._preprocess_chat.assert_not_called()


def test_render_engine_dead_raises():
    handler = _make_render_handler()
    handler.engine_client.errored = True
    handler.engine_client.dead_error = RuntimeError("Engine dead")
    with pytest.raises(RuntimeError, match="Engine dead"):
        asyncio.run(handler.render_batch_chat_request(_make_batch_request(3)))


def test_render_template_validation_called_once():
    handler = _make_render_handler()
    asyncio.run(handler.render_batch_chat_request(_make_batch_request(4)))
    handler.online_renderer.validate_chat_template.assert_called_once()


def test_render_template_validation_error_propagates():
    handler = _make_render_handler()
    handler.online_renderer.validate_chat_template.return_value = ErrorResponse(
        error=ErrorInfo(message="untrusted", type="BadRequestError", code=400),
    )
    result = asyncio.run(handler.render_batch_chat_request(_make_batch_request(3)))
    assert isinstance(result, ErrorResponse)
    handler._preprocess_chat.assert_not_called()


def test_render_returns_correct_count():
    handler = _make_render_handler()
    result = asyncio.run(handler.render_batch_chat_request(_make_batch_request(4)))
    assert not isinstance(result, ErrorResponse)
    conversations, prompts, single_requests = result
    assert len(conversations) == 4
    assert len(prompts) == 4
    assert len(single_requests) == 4


def test_render_preprocess_called_per_item():
    handler = _make_render_handler()
    asyncio.run(handler.render_batch_chat_request(_make_batch_request(3)))
    assert handler._preprocess_chat.call_count == 3


def test_render_audio_format_error_propagates():
    handler = _make_render_handler()
    handler.engine_client.output_modalities = ["text", "audio"]
    request = _make_batch_request(2)
    request.modalities = ["text", "audio"]
    handler._resolve_audio_format.return_value = ErrorResponse(
        error=ErrorInfo(message="bad format", type="BadRequestError", code=400),
    )
    result = asyncio.run(handler.render_batch_chat_request(request))
    assert isinstance(result, ErrorResponse)
