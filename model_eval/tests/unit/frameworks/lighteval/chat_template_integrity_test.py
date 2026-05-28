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

"""Chat-template integrity and client-server wire-format tests for the Lighteval adapter.

Asserts that the framework adapter transmits raw, un-templated message lists to
the `/v1/chat/score` server endpoint instead of applying prompt flattening on
the client. This ensures correct multi-turn and few-shot history processing at
the server boundary.
"""

import json
from unittest import mock

from absl.testing import absltest

import pytest

pytest.importorskip(
    "lighteval",
    reason='install via `pip install -e ".[lighteval]"` to run lighteval tests',
)

# pylint: disable=g-import-not-at-top,g-bad-import-order
from lighteval.models.endpoints import litellm_model

from model_eval.frameworks.lighteval import _chat_score_lighteval_model

# Chat-template tokens we expect to NEVER appear in the wire payload's
# `content` fields. If they do, the framework has flattened a templated
# string into the user turn (or some upstream layer has) and the server will
# template it again, producing double-wrapped prompts.
_CHAT_TEMPLATE_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<start_of_turn>",
    "<end_of_turn>",
    "[INST]",
    "[/INST]",
)


class _FakeDoc:
  """Minimal stand-in for the `Doc` lighteval passes to `loglikelihood`."""

  def __init__(self, query, choices):
    self.query = query
    self.choices = choices


class _FakeJsonChatStr:
  """Mimics lighteval/lm-eval's `JsonChatStr` — exposes `.prompt` as JSON.

  This shape is what lighteval produces when a task config requests
  `apply_chat_template` at the *task* layer; the resulting wrapper carries
  a structured chat history (a JSON-serialized list of role/content dicts),
  NOT a templated flat string.
  """

  def __init__(self, messages_list):
    self.prompt = json.dumps(messages_list)


def _build_model_with_recording_client(captured, response=None):
  """Constructs `ChatScoreLightevalModel` with `httpx.Client` patched to record requests."""
  if response is None:
    response = {"choices": [{"score": -1.5, "logprobs": {"is_greedy": False}}]}

  mock_response = mock.MagicMock()
  mock_response.json.return_value = response
  mock_response.raise_for_status.return_value = None

  mock_client = mock.MagicMock()
  mock_client.__enter__.return_value = mock_client
  mock_client.__exit__.return_value = None

  def _post(url, json=None):  # pylint: disable=redefined-outer-name
    captured.append({"url": url, "json": json})
    return mock_response

  mock_client.post.side_effect = _post

  config = litellm_model.LiteLLMModelConfig(
      model_name="openai/test-model",
      api_key="not-needed",
      base_url="http://127.0.0.1:8080/v1",
      concurrent_requests=1,
  )
  with mock.patch.object(
      _chat_score_lighteval_model.httpx, "Client", return_value=mock_client
  ):
    return _chat_score_lighteval_model.ChatScoreLightevalModel(config)


class ChatTemplateIntegrityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.captured = []

  def _model(self):
    return _build_model_with_recording_client(self.captured)

  def _patched_httpx(self, model):
    """Returns a patcher that re-binds the recording client into the loglikelihood call."""
    del model  # Unused.
    mock_response = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"score": -1.5, "logprobs": {"is_greedy": False}}]
    }
    mock_response.raise_for_status.return_value = None

    mock_client = mock.MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    def _post(url, json=None):  # pylint: disable=redefined-outer-name
      self.captured.append({"url": url, "json": json})
      return mock_response

    mock_client.post.side_effect = _post
    return mock.patch.object(
        _chat_score_lighteval_model.httpx, "Client", return_value=mock_client
    )

  def test_plain_context_wraps_as_single_user_turn(self):
    """A plain-string `Doc.query` becomes exactly one user turn, no template tokens."""
    model = self._model()
    doc = _FakeDoc(
        query="Question: What is the capital of France?\nAnswer:",
        choices=[" Paris", " London"],
    )
    with self._patched_httpx(model):
      model.loglikelihood([doc])

    # Two POSTs (one per choice), each to the score endpoint.
    self.assertEqual(len(self.captured), 2)
    for i, req in enumerate(self.captured):
      self.assertEqual(req["url"], "http://127.0.0.1:8080/v1/chat/score")
      messages = req["json"]["messages"]
      self.assertEqual(
          len(messages), 2, msg=f"choice {i}: expected exactly 2 messages"
      )
      self.assertEqual(messages[0]["role"], "user")
      self.assertEqual(messages[-1]["role"], "assistant")
      self.assertEqual(messages[0]["content"], doc.query)
      for token in _CHAT_TEMPLATE_TOKENS:
        self.assertNotIn(
            token,
            messages[0]["content"],
            msg=(
                f"choice {i}: chat-template token {token!r} leaked into user"
                " content — framework is double-templating, server will"
                " template again."
            ),
        )

  def test_jsonchatstr_context_is_unpacked_verbatim(self):
    """Structured chat history is passed through as a message list, not flattened."""
    chat_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second question?"},
    ]
    model = self._model()
    doc = _FakeDoc(
        query=_FakeJsonChatStr(chat_history), choices=[" The answer."]
    )
    with self._patched_httpx(model):
      model.loglikelihood([doc])

    self.assertEqual(len(self.captured), 1)
    messages = self.captured[0]["json"]["messages"]
    # Original 4 messages preserved verbatim + the assistant continuation.
    self.assertEqual(len(messages), 5)
    for orig, sent in zip(chat_history, messages[:4]):
      self.assertEqual(orig, sent)
    self.assertEqual(
        messages[-1], {"role": "assistant", "content": " The answer."}
    )
    # Critically: no role's content should contain chat-template tokens. If it
    # does, upstream flattened the chat into a templated string before
    # wrapping it as JsonChatStr — which the server would then re-template.
    for msg in messages:
      content = msg["content"]
      if not isinstance(content, str):
        continue
      for token in _CHAT_TEMPLATE_TOKENS:
        self.assertNotIn(
            token,
            content,
            msg=f"chat-template token {token!r} leaked into message {msg!r}",
        )

  def test_continuation_is_only_in_assistant_role(self):
    """Each choice produces one request with the continuation in the assistant role only."""
    model = self._model()
    choices = [" alpha", " beta", " gamma"]
    doc = _FakeDoc(query="prefix", choices=choices)
    with self._patched_httpx(model):
      model.loglikelihood([doc])

    self.assertEqual(len(self.captured), len(choices))
    for choice, req in zip(choices, self.captured):
      messages = req["json"]["messages"]
      self.assertEqual(messages[-1], {"role": "assistant", "content": choice})
      # The continuation must not also be concatenated into the user content.
      self.assertNotIn(choice, messages[0]["content"])

  def test_literal_special_tokens_in_content_pass_through(self):
    """If the task content legitimately contains literal special-token strings, pass them verbatim."""
    weird_query = (
        "Explain what <|im_end|> means in a transformer chat template."
    )
    model = self._model()
    doc = _FakeDoc(query=weird_query, choices=[" It marks the end of a turn."])
    with self._patched_httpx(model):
      model.loglikelihood([doc])

    self.assertEqual(len(self.captured), 1)
    messages = self.captured[0]["json"]["messages"]
    # Framework must pass through verbatim — escaping is the server's job.
    self.assertEqual(messages[0]["content"], weird_query)
    # Continuation untouched too.
    self.assertEqual(messages[-1]["content"], " It marks the end of a turn.")

  def test_empty_context_logs_warning_and_sends_only_assistant_turn(self):
    """Empty `Doc.query` produces a single-message payload (assistant continuation only).

    This documents current behavior; whether this is what the server expects is
    a separate question that surfaces only in end-to-end runs. The litert-lm
    server raises HTTP 400 if there are fewer than 2 messages — so this case,
    if it ever arises in practice, would fail at the server boundary, not
    silently produce wrong scores.
    """
    model = self._model()
    doc = _FakeDoc(query="", choices=[" hello"])
    with self._patched_httpx(model):
      model.loglikelihood([doc])

    self.assertEqual(len(self.captured), 1)
    messages = self.captured[0]["json"]["messages"]
    # Empty context → no user turn was appended; only the assistant
    # continuation.
    self.assertEqual(messages, [{"role": "assistant", "content": " hello"}])

  def test_score_url_strips_v1_correctly(self):
    """The score URL is constructed by stripping the trailing `/v1` from base_url."""
    model = self._model()
    self.assertEqual(model._score_url, "http://127.0.0.1:8080/v1/chat/score")


if __name__ == "__main__":
  absltest.main()
