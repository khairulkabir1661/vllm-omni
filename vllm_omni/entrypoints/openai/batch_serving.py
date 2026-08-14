# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import Request
from pydantic import ValidationError
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    UsageInfo,
)
from vllm.logger import init_logger
from vllm.tokenizers.mistral import (
    MistralTokenizer,
    maybe_serialize_tool_calls,
    truncate_tool_call_ids,
    validate_request_params,
)

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.utils import is_single_stage_diffusion

logger = init_logger(__name__)


class OmniOpenAIServingChatBatch(OmniOpenAIServingChat):
    @staticmethod
    def _maybe_collapse_choices(choices: list[ChatCompletionResponseChoice]):
        """Collapse content + audio choices into one choice.

        NOTE: This is a hack for models that produce separate text and
        audio choices to ensure we have 1:1 mapping between number of
        inputs in the batch and responses.
        """
        content_choice, audio_choice = None, None
        if len(choices) == 1:
            return choices[0]
        if len(choices) != 2:
            raise ValueError(f"Expected 1 or 2 choices to collapse, got {len(choices)}. ")

        for choice in choices:
            if choice.message.audio:
                if audio_choice is not None:
                    raise ValueError("Multiple audio choices cannot be set in one completion message")
                audio_choice = choice
            else:
                if content_choice is not None:
                    raise ValueError("Multiple content choices cannot be set in one completion message")
                content_choice = choice

        if content_choice is None or audio_choice is None:
            raise ValueError(
                "Expected one content and one audio choice, but got: "
                f"content={content_choice is not None}, audio={audio_choice is not None}"
            )
        content_choice.message.audio = audio_choice.message.audio
        return content_choice

    @staticmethod
    def _get_subrequest_ids(base_id, request: BatchChatCompletionRequest) -> list[str]:
        """Get the request ID for each entry in the batch request to be resubmitted to chat completions."""
        return [f"{base_id}-idx-{idx}" for idx in range(len(request.messages))]

    async def render_batch_chat_request(
        self,
        request: BatchChatCompletionRequest,
    ) -> tuple[list[Any], list[Any], list[ChatCompletionRequest]] | ErrorResponse:
        """Validate the batch request once and preprocess each conversation.

        Runs all shared checks (model, engine health, LoRA, tokenizer,
        parsers, template, modalities, audio format) a single time,
        then preprocesses each conversation individually.

        Returns:
            (all_conversations, all_engine_prompts, single_requests) on
            success, or ErrorResponse on failure.
        """
        # --- Checks that only need to run once for the whole batch ---

        # Check 1: Model existence
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # Check 2: Engine health
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        # Check 3: LoRA adapter
        self._maybe_get_adapters(request, supports_default_mm_loras=True)

        # Check 5: Tokenizer
        renderer = self.renderer
        tokenizer = renderer.get_tokenizer()
        if tokenizer is None:
            tokenizer = await self.engine_client.get_tokenizer()

        # Check 6: Reasoning parser
        if self.parser_cls is not None and self.parser_cls.reasoning_parser_cls is not None:
            self._effective_chat_template_kwargs(request)

        # Check 7: Tool parser class
        tool_parser = self.parser_cls.tool_parser_cls if self.parser_cls is not None else None

        # Check 8: Mistral tokenizer special handling
        if isinstance(tokenizer, MistralTokenizer):
            maybe_serialize_tool_calls(request)
            truncate_tool_call_ids(request)
            validate_request_params(request)

        # Check 9: Tool choice validation + tool_dicts
        tool_choice = getattr(request, "tool_choice", None)
        tools = getattr(request, "tools", None)
        tool_parsing_unavailable = (
            tool_parser is None and not isinstance(tokenizer, MistralTokenizer) and not self.use_harmony
        )
        if tool_parsing_unavailable and tool_choice not in (None, "none"):
            if tool_choice == "auto" and not self.enable_auto_tools:
                return self.create_error_response(
                    '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
                )
            elif tool_choice != "auto":
                return self.create_error_response(f'tool_choice="{tool_choice}" requires --tool-call-parser to be set')

        if tools is None or (tool_choice == "none" and self.exclude_tools_when_tool_choice_none):
            tool_dicts = None
        else:
            tool_dicts = [tool.model_dump() for tool in tools]

        # Check 10: Chat template validation
        if not self.use_harmony:
            error_check_ret = self.online_renderer.validate_chat_template(
                request_chat_template=getattr(request, "chat_template", None),
                chat_template_kwargs=getattr(request, "chat_template_kwargs", None),
                trust_request_chat_template=self.trust_request_chat_template,
            )
            if error_check_ret is not None:
                return error_check_ret

        # Check 11: Output modalities validation
        engine_output_modalities = [x for x in self.engine_client.output_modalities if x is not None]
        output_modalities = getattr(request, "modalities", engine_output_modalities)
        request.modalities = output_modalities if output_modalities is not None else engine_output_modalities

        if not isinstance(request.modalities, list) or not all(isinstance(m, str) for m in request.modalities):
            return self.create_error_response("'modalities' must be a list of strings.")
        allowed_modalities = set(engine_output_modalities)
        if is_single_stage_diffusion(self.engine_client):
            allowed_modalities.add("text")
        unsupported = set(request.modalities) - allowed_modalities
        if unsupported:
            return self.create_error_response(
                f"Unsupported output modalities {', '.join(sorted(unsupported))} "
                f"for this model. Supported modalities: "
                f"{', '.join(sorted(allowed_modalities))}",
            )

        # Check 12: Audio format validation
        if request.modalities and "audio" in request.modalities:
            audio_format_check = self._resolve_audio_format(request)
            if isinstance(audio_format_check, ErrorResponse):
                return audio_format_check

        # --- Per-item preprocessing ---

        all_conversations: list[Any] = []
        all_engine_prompts: list[Any] = []
        single_requests: list[ChatCompletionRequest] = []

        for messages in request.messages:
            single_request = request.to_chat_completion_request(messages)
            single_request.stream = False
            single_requests.append(single_request)

            try:
                if not self.use_harmony:
                    merged_kwargs = self._effective_chat_template_kwargs(single_request)
                    conversation, engine_prompts = await self._preprocess_chat(
                        single_request,
                        single_request.messages,
                        default_template=(single_request.chat_template or self.chat_template),
                        default_template_content_format=(self.chat_template_content_format),
                        default_template_kwargs=merged_kwargs,
                        tool_dicts=tool_dicts,
                        tool_parser=tool_parser,
                        renderer=renderer,
                        add_generation_prompt=single_request.add_generation_prompt,
                        continue_final_message=single_request.continue_final_message,
                        documents=getattr(single_request, "documents", None),
                        add_special_tokens=single_request.add_special_tokens,
                    )
                else:
                    should_include_tools = tool_dicts is not None
                    conversation, engine_prompts = self.online_renderer._make_request_with_harmony(
                        single_request,
                        should_include_tools,
                    )
            except (ValueError, TypeError, RuntimeError) as e:
                logger.exception("Error preprocessing batch item")
                message = str(e)
                if e.__cause__ is not None:
                    message = f"{message} {e.__cause__}"
                return self.create_error_response(message)

            all_conversations.append(conversation)
            all_engine_prompts.append(engine_prompts[0])

        return all_conversations, all_engine_prompts, single_requests

    async def create_batch_chat_completion(
        self,
        request: BatchChatCompletionRequest,
        raw_request: Request,
    ) -> ChatCompletionResponse | ErrorResponse:
        """Given a request, submit each request to chat completions & collect the results."""
        return await self._create_batch_chat_completion_legacy(request, raw_request)

    async def _create_batch_chat_completion_legacy(
        self,
        request: BatchChatCompletionRequest,
        raw_request: Request,
    ) -> ChatCompletionResponse | ErrorResponse:
        """Legacy implementation: submits each item via create_chat_completion."""
        model = ""
        enabled_streaming = False
        chat_requests: list[ChatCompletionRequest] = []
        choices: list[ChatCompletionResponseChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Get the base request from the raw_request before maybe mutating the header
        base_id = OmniOpenAIServingChatBatch._base_request_id(
            raw_request,
            request.request_id,
        )
        batch_req_id = f"chatcmpl-batch-{base_id}"
        sub_req_ids = self._get_subrequest_ids(base_id, request)

        # Remove the raw request header if it exists now that we have subreq IDs,
        # since this takes priority over the object request_ids
        if raw_request.headers.get("X-Request-Id"):
            copy_headers = raw_request.headers.mutablecopy()
            del copy_headers["X-Request-Id"]
            raw_request._headers = copy_headers

        for idx, msg in enumerate(request.messages):
            # Update the current request ID to avoid subrequest collisions
            request.request_id = sub_req_ids[idx]
            try:
                chat_cmp_request = request.to_chat_completion_request(msg)
            except ValidationError as e:
                return self._create_error_response(
                    f"Message {idx} could not be converted to a chat completion request: {e}",
                )
            # Streaming isn't supported for batch chat completions,
            # so always set to False & warn if it was requested.
            enabled_streaming |= chat_cmp_request.stream
            chat_cmp_request.stream = False
            chat_requests.append(chat_cmp_request)

        if enabled_streaming:
            logger.warning("Streaming is not supported for batched chat completions; ignoring stream=True.")

        # Submit each chat completion request as a task, then gather results.
        # TODO (Alex): optimize this
        tasks = [asyncio.create_task(self.create_chat_completion(c, raw_request)) for c in chat_requests]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            # Ensure we cancel remaining tasks if needed, e.g., early exit due to bad behavior
            for t in tasks:
                if not t.done():
                    t.cancel()

        for i, resp in enumerate(results):
            if isinstance(resp, ErrorResponse):
                return resp
            completion: ChatCompletionResponse = resp
            model = completion.model
            # FIXME (Alex): We should probably handle this in chat completions,
            # not here, but we need to ensure streaming is properly handled
            try:
                collapsed_choice = self._maybe_collapse_choices(completion.choices)
            except ValueError as e:
                return self._create_error_response(
                    f"Failed to collapse choices with error: {e}",
                )
            collapsed_choice.index = i
            choices.append(collapsed_choice)
            if completion.usage:
                total_prompt_tokens += completion.usage.prompt_tokens
                total_completion_tokens += completion.usage.completion_tokens

        usage = UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        )

        return ChatCompletionResponse(
            id=batch_req_id,
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
        )
