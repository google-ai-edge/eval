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


if __name__ == "__main__":
  unittest.main()
