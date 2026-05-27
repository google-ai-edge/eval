# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for LiteRT LM server."""

import unittest
from unittest import mock

from model_eval.runners.litert_lm import _litert_lm_server
import fastapi.testclient


class TestLiteRTLMServer(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_engine = mock.MagicMock()
    mock_cfg = mock.MagicMock()
    mock_cfg.model_name = "test-model"
    mock_cfg.always_return_not_greedy = False
    self.app = _litert_lm_server.build_app(self.mock_engine, mock_cfg)
    self.client = fastapi.testclient.TestClient(self.app)

  def test_health(self):
    response = self.client.get("/health")
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json(), {"status": "ok"})

  def test_chat_completions_sampling_shape(self):
    mock_conv = mock.MagicMock()
    self.mock_engine.create_conversation.return_value.__enter__.return_value = (
        mock_conv
    )

    mock_conv.send_message.return_value = {
        "role": "model",
        "content": [{"type": "text", "text": "Hi there!"}],
    }

    response = self.client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertEqual(data["model"], "test-model")
    self.assertEqual(data["object"], "chat.completion")

    choices = data["choices"]
    self.assertEqual(len(choices), 1)
    self.assertEqual(choices[0]["message"]["content"], "Hi there!")

  @mock.patch(
      "model_eval.runners.litert_lm._litert_lm_server.litert_lm.SamplerConfig"
  )
  def test_chat_completions_sampler_config(self, mock_sampler_config_class):
    mock_conv = mock.MagicMock()
    self.mock_engine.create_conversation.return_value.__enter__.return_value = (
        mock_conv
    )
    mock_sampler_config_instance = mock.MagicMock()
    mock_sampler_config_class.return_value = mock_sampler_config_instance
    mock_conv.send_message.return_value = {
        "role": "model",
        "content": [{"type": "text", "text": "Testing temperature"}],
    }

    response = self.client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
        },
    )

    self.assertEqual(response.status_code, 200)
    mock_sampler_config_class.assert_called_once_with(temperature=0.8)
    self.mock_engine.create_conversation.assert_called_once_with(
        messages=None, sampler_config=mock_sampler_config_instance
    )

  def test_chat_completions_stop_sequence(self):
    mock_conv = mock.MagicMock()
    self.mock_engine.create_conversation.return_value.__enter__.return_value = (
        mock_conv
    )

    mock_conv.send_message.return_value = {
        "role": "model",
        "content": [{"type": "text", "text": "Hello world, how are you?"}],
    }

    response = self.client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stop": ["world"],
        },
    )

    self.assertEqual(response.status_code, 200)
    data = response.json()
    choices = data["choices"]
    self.assertEqual(len(choices), 1)
    self.assertEqual(choices[0]["message"]["content"], "Hello ")

  def test_mm_content_part_translation(self):
    msg = _litert_lm_server._ChatMessage(
        role="user",
        content=[{"type": "text", "text": "A"}],
    )
    result = _litert_lm_server._to_litert_lm_message(msg)
    self.assertEqual(result["role"], "user")
    self.assertEqual(result["content"], [{"type": "text", "text": "A"}])

    msg_image = _litert_lm_server._ChatMessage(
        role="user",
        content=[{
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0"},
        }],
    )
    result_image = _litert_lm_server._to_litert_lm_message(msg_image)
    self.assertEqual(result_image["role"], "user")
    self.assertEqual(
        result_image["content"], [{"type": "image", "blob": "iVBORw0"}]
    )

    msg_audio = _litert_lm_server._ChatMessage(
        role="user",
        content=[
            {"type": "input_audio", "input_audio": {"data": "base64audio"}}
        ],
    )
    result_audio = _litert_lm_server._to_litert_lm_message(msg_audio)
    self.assertEqual(result_audio["role"], "user")
    self.assertEqual(
        result_audio["content"], [{"type": "audio", "blob": "base64audio"}]
    )

  def test_mm_content_part_translation_invalid_image(self):
    msg_image_invalid = _litert_lm_server._ChatMessage(
        role="user",
        content=[{
            "type": "image_url",
            "image_url": {"url": "http://example.com/image.png"},
        }],
    )
    with self.assertRaises(fastapi.HTTPException) as cm:
      _litert_lm_server._to_litert_lm_message(msg_image_invalid)
    self.assertEqual(cm.exception.status_code, 400)
    self.assertIn("HTTP image URLs are not supported", cm.exception.detail)

  def test_chat_score_valid(self):
    mock_session = mock.MagicMock()
    self.mock_engine.create_session.return_value.__enter__.return_value = (
        mock_session
    )

    mock_conversation = mock.MagicMock()
    self.mock_engine.create_conversation.return_value.__enter__.return_value = (
        mock_conversation
    )
    mock_conversation.render_message_to_string.return_value = "Context Applied"

    # Mock the return value of run_text_scoring
    mock_result = mock.MagicMock()
    mock_result.scores = [-10.5]
    mock_result.token_scores = [[-2.0, -8.5]]
    mock_session.run_text_scoring.return_value = mock_result

    # Mock the session run_decode for is_greedy check
    mock_decode_responses = mock.MagicMock()
    mock_decode_responses.texts = ["Hi there, world"]
    mock_session.run_decode.return_value = mock_decode_responses

    response = self.client.post(
        "/v1/chat/score",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        },
    )

    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertEqual(data["model"], "test-model")
    self.assertEqual(data["object"], "chat.score")

    choices = data["choices"]
    self.assertEqual(len(choices), 1)

    score_data = choices[0]
    self.assertEqual(score_data["score"], -10.5)

    logprobs = score_data["logprobs"]
    self.assertNotIn("tokens", logprobs)
    self.assertEqual(logprobs["token_logprobs"], [-2.0, -8.5])
    self.assertTrue(logprobs["is_greedy"])
    self.assertNotIn("top_logprobs", logprobs)

    self.assertEqual(mock_session.run_prefill.call_count, 2)
    mock_session.run_text_scoring.assert_called_once_with(["Hi there"])
    mock_session.run_decode.assert_called_once()

  def test_chat_score_only_one_message(self):
    response = self.client.post(
        "/v1/chat/score",
        json={
            "model": "test-model",
            "messages": [{"role": "assistant", "content": "Hi there"}],
        },
    )
    self.assertEqual(response.status_code, 400)
    self.assertIn("at least one context message", response.json()["detail"])

  def test_chat_score_last_message_not_assistant(self):
    response = self.client.post(
        "/v1/chat/score",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "Are you there?"},
            ],
        },
    )
    self.assertEqual(response.status_code, 400)
    self.assertIn("must be role=assistant", response.json()["detail"])

  def test_chat_score_continuation_not_string(self):
    response = self.client.post(
        "/v1/chat/score",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi"}],
                },
            ],
        },
    )
    self.assertEqual(response.status_code, 400)
    self.assertIn("must be a plain string", response.json()["detail"])

  def test_chat_score_always_return_not_greedy(self):
    """Verifies that setting always_return_not_greedy dynamically skips the full sequence decoding path completely."""
    mock_session = mock.MagicMock()
    self.mock_engine.create_session.return_value.__enter__.return_value = (
        mock_session
    )

    mock_conversation = mock.MagicMock()
    self.mock_engine.create_conversation.return_value.__enter__.return_value = (
        mock_conversation
    )
    mock_conversation.render_message_to_string.return_value = "Context Applied"

    mock_result = mock.MagicMock()
    mock_result.scores = [-5.0]
    mock_result.token_scores = [[-2.5, -2.5]]
    mock_session.run_text_scoring.return_value = mock_result

    mock_cfg = mock.MagicMock()
    mock_cfg.model_name = "test-model"
    mock_cfg.always_return_not_greedy = True
    app_fast = _litert_lm_server.build_app(self.mock_engine, mock_cfg)
    client_fast = fastapi.testclient.TestClient(app_fast)

    response = client_fast.post(
        "/v1/chat/score",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        },
    )

    self.assertEqual(response.status_code, 200)
    data = response.json()
    logprobs = data["choices"][0]["logprobs"]
    self.assertEqual(logprobs["token_logprobs"], [-2.5, -2.5])
    self.assertFalse(logprobs["is_greedy"])
    # Decoding step should be completely skipped!
    mock_session.run_decode.assert_not_called()

  def test_chat_completions_accepts_max_tokens_without_error(self):
    """Schema-acceptance regression test.

    The underlying `Conversation` API has no per-call token budget hook,
    so `max_tokens` is NOT enforced by this endpoint. But the wire format
    must still be valid: lighteval / litellm send `max_tokens` (and the
    newer-spec alias `max_completion_tokens`) on every generation request, and
    pydantic must accept both without 422'ing the request. This test guards
    against a future schema change that drops those fields.
    """
    mock_conv = mock.MagicMock()
    self.mock_engine.create_conversation.return_value.__enter__.return_value = (
        mock_conv
    )
    mock_conv.send_message.return_value = {
        "role": "model",
        "content": [{"type": "text", "text": "ok"}],
    }
    for body in (
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 128,
        },
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 128,
        },
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 64,
            "max_completion_tokens": 128,
        },
    ):
      response = self.client.post("/v1/chat/completions", json=body)
      self.assertEqual(
          response.status_code,
          200,
          msg=(
              f"request body {body!r} was rejected (status "
              f"{response.status_code}). pydantic must accept "
              "max_tokens and max_completion_tokens even though the "
              "endpoint does not enforce them."
          ),
      )


class TestRenderChatScoreContext(unittest.TestCase):
  """Regression tests for `_render_chat_score_context`.

  Before the fix, `_chat_score` rendered the conversation by calling
  `conversation.render_message_to_string(m)` once per message and
  concatenating. That produces structurally invalid prompts for any context
  with more than one message — the engine template treats every render as
  the final message in the conversation. After the fix, the helper preloads
  all-but-the-last message via `create_conversation(messages=...)` and
  renders only the final message, so the engine templates the full
  conversation correctly with one generation-prompt suffix at the end.

  These tests verify the call shape (mocks), not real template output —
  real-engine validation is in `scripts/probe_prompt_parity.py`.
  """

  def _build_engine_mock(self, render_return="rendered"):
    engine = mock.MagicMock()
    conv = mock.MagicMock()
    engine.create_conversation.return_value.__enter__.return_value = conv
    conv.render_message_to_string.return_value = render_return
    return engine, conv

  def test_empty_context_returns_empty_string(self):
    engine, _ = self._build_engine_mock()
    out = _litert_lm_server._render_chat_score_context(engine, [])
    self.assertEqual(out, "")
    engine.create_conversation.assert_not_called()

  def test_single_message_preloads_empty_renders_one(self):
    """One-message context: preload [] and render that single message."""
    engine, conv = self._build_engine_mock()
    msg = {"role": "user", "content": "hello"}
    out = _litert_lm_server._render_chat_score_context(engine, [msg])
    self.assertEqual(out, "rendered")
    # All-but-last is empty when there's only one message.
    engine.create_conversation.assert_called_once_with(messages=[])
    conv.render_message_to_string.assert_called_once_with(msg)

  def test_multi_message_preloads_all_but_last(self):
    """Three-message context: preload [m1, m2, m3] and render m4 (the fix)."""
    engine, conv = self._build_engine_mock()
    m1 = {"role": "system", "content": "sys"}
    m2 = {"role": "user", "content": "hi"}
    m3 = {"role": "assistant", "content": "hello"}
    m4 = {"role": "user", "content": "q2"}
    _litert_lm_server._render_chat_score_context(engine, [m1, m2, m3, m4])
    call_kwargs = engine.create_conversation.call_args.kwargs
    preloaded = call_kwargs["messages"]
    self.assertEqual(len(preloaded), 3)
    self.assertEqual(preloaded[0], {"role": "system", "content": "sys"})
    self.assertEqual(preloaded[1], {"role": "user", "content": "hi"})
    self.assertEqual(preloaded[2], {"role": "assistant", "content": "hello"})
    rendered = conv.render_message_to_string.call_args.args[0]
    self.assertEqual(rendered, {"role": "user", "content": "q2"})

  def test_messages_passed_verbatim(self):
    """Adapter must not rewrite role names; the template is the right layer."""
    engine, conv = self._build_engine_mock()
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    _litert_lm_server._render_chat_score_context(engine, msgs)
    preloaded = engine.create_conversation.call_args.kwargs["messages"]
    self.assertEqual(
        preloaded,
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
        msg="role names must reach the engine unchanged",
    )

  def test_input_is_not_mutated(self):
    """Caller's message dicts must not be modified in-place."""
    engine, _ = self._build_engine_mock()
    msg_in = {"role": "assistant", "content": "hi"}
    msgs = [msg_in, {"role": "user", "content": "q"}]
    _litert_lm_server._render_chat_score_context(engine, msgs)
    self.assertEqual(
        msg_in,
        {"role": "assistant", "content": "hi"},
        msg="caller's dict was mutated",
    )

  def test_render_per_message_is_not_called(self):
    """Smoke check: the broken pattern (render per message + concat) is gone.

    If a future refactor re-introduces `[render(m) for m in msgs]`, this
    will fail by detecting more than one render call.
    """
    engine, conv = self._build_engine_mock()
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    _litert_lm_server._render_chat_score_context(engine, msgs)
    self.assertEqual(
        conv.render_message_to_string.call_count,
        1,
        msg=(
            "render_message_to_string was called more than once — the"
            " per-message+concat pattern appears to have been reintroduced."
        ),
    )


if __name__ == "__main__":
  unittest.main()
