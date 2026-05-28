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

"""Chat-template integrity and client-server wire-format tests for the LM Eval framework adapter.

Exercises `LocalChatScoreModel.loglikelihood` to ensure that context packaged as
a `JsonChatStr` (JSON-serialized chat history) is cleanly extracted, unwrapped,
and transmitted as a raw message list to `/v1/chat/score`, properly preserving
few-shot history format.
"""

import json
from unittest import mock

from absl.testing import absltest

import pytest

pytest.importorskip(
    "lm_eval",
    reason=(
        'install via `pip install -e ".[lighteval]"` for the JsonChatStr type,'
        " or any base install"
    ),
)

# pylint: disable=g-import-not-at-top,g-bad-import-order
from lm_eval.models import api_models
from model_eval.frameworks import _local_chat_score_model

# Chat-template tokens we expect to NEVER appear in the wire payload's
# `content` fields. If they do, lm_eval upstream started flattening
# templated text into the chat history, or our adapter started doing
# template-rendering it shouldn't.
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


class _FakeRequest:
  """Minimal stand-in for lm_eval's `Instance`-like request object.

  `LocalChatScoreModel.loglikelihood` accesses `request.args` which
  is a `(context, continuation)` tuple for scoring requests, or a
  longer tuple with additional content for multimodal.
  """

  def __init__(self, context, continuation):
    self.args = (context, continuation)


def _build_model_with_recording_post(captured, response=None):
  """Build a `LocalChatScoreModel` with `requests_lib.post` patched to record requests.

  Bypasses the heavy `LocalChatCompletion.__init__` via `__new__` —
  we only need `_scoring_url` and `model` set for the loglikelihood
  call. Patching is at the symbol our module imported it under
  (`requests_lib`), not at `requests.post` globally.

  Args:
    captured: A list into which the captured URL and payload for each POST
      request will be appended.
    response: An optional dictionary representing the JSON response payload to
      be returned by the mocked POST endpoint. Defaults to a mock success
      response.

  Returns:
    A tuple `(model, fake_post)` containing the uninitialized
    `LocalChatScoreModel` instance and the mocked `requests_lib.post`
    side-effect function.
  """
  if response is None:
    response = {"choices": [{"score": -1.5, "logprobs": {"is_greedy": False}}]}

  mock_resp = mock.MagicMock()
  mock_resp.json.return_value = response
  mock_resp.raise_for_status.return_value = None

  def fake_post(url, json=None, **kw):  # pylint: disable=redefined-outer-name
    del kw  # Unused.
    captured.append({"url": url, "json": json})
    return mock_resp

  model = _local_chat_score_model.LocalChatScoreModel.__new__(
      _local_chat_score_model.LocalChatScoreModel
  )
  model._scoring_url = "http://127.0.0.1:8080/v1/chat/score"
  model.model = "test-model"
  return model, fake_post


class LmEvalChatTemplateIntegrityTest(absltest.TestCase):
  """Regression suite for the lm_eval → server wire format."""

  def setUp(self):
    super().setUp()
    self.captured = []

  def _model_and_patcher(self):
    model, fake_post = _build_model_with_recording_post(self.captured)
    return model, mock.patch(
        "model_eval.frameworks._local_chat_score_model.requests_lib.post",
        side_effect=fake_post,
    )

  def test_plain_string_context_wraps_as_single_user_turn(self):
    """Plain-string context → single user turn, no template tokens."""
    model, patcher = self._model_and_patcher()
    req = _FakeRequest(
        context="Question: What is the capital of France?\nAnswer:",
        continuation=" Paris",
    )
    with patcher:
      model.loglikelihood([req], disable_tqdm=True)

    self.assertLen(self.captured, 1)
    msgs = self.captured[0]["json"]["messages"]
    self.assertEqual(
        msgs,
        [
            {
                "role": "user",
                "content": "Question: What is the capital of France?\nAnswer:",
            },
            {"role": "assistant", "content": " Paris"},
        ],
    )
    for tok in _CHAT_TEMPLATE_TOKENS:
      self.assertNotIn(tok, msgs[0]["content"])

  def test_jsonchatstr_few_shot_history_unpacked_verbatim(self):
    """Few-shot history must reach server as a clean message list.

    This test ensures the client side puts the right shape on the wire
    so the server's now-correct render gets the right input.
    """
    chat_history = [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Q: 1+1?\nA:"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "Q: 2+2?\nA:"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Q: 3+3?\nA:"},
    ]
    model, patcher = self._model_and_patcher()
    req = _FakeRequest(
        context=api_models.JsonChatStr(json.dumps(chat_history)),
        continuation=" 6",
    )
    with patcher:
      model.loglikelihood([req], disable_tqdm=True)

    self.assertLen(self.captured, 1)
    msgs = self.captured[0]["json"]["messages"]
    expected = chat_history + [{"role": "assistant", "content": " 6"}]
    self.assertEqual(msgs, expected)
    # No role's content should contain chat-template tokens. If any do,
    # lm_eval upstream started templating before sending — investigate.
    for msg in msgs:
      content = msg["content"]
      if not isinstance(content, str):
        continue
      for tok in _CHAT_TEMPLATE_TOKENS:
        self.assertNotIn(
            tok,
            content,
            msg=f"template token {tok!r} leaked into {msg['role']!r} content",
        )

  def test_jsonchatstr_system_plus_user_unpacked_verbatim(self):
    """Two-message system+user case."""
    chat_history = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Q?"},
    ]
    model, patcher = self._model_and_patcher()
    req = _FakeRequest(
        context=api_models.JsonChatStr(json.dumps(chat_history)),
        continuation=" A.",
    )
    with patcher:
      model.loglikelihood([req], disable_tqdm=True)

    msgs = self.captured[0]["json"]["messages"]
    self.assertEqual(
        msgs,
        chat_history + [{"role": "assistant", "content": " A."}],
    )

  def test_continuation_is_only_in_assistant_role(self):
    """The continuation lands exclusively in the assistant role at the end."""
    model, patcher = self._model_and_patcher()
    req = _FakeRequest(context="prefix", continuation=" unique-marker")
    with patcher:
      model.loglikelihood([req], disable_tqdm=True)

    msgs = self.captured[0]["json"]["messages"]
    self.assertEqual(
        msgs[-1], {"role": "assistant", "content": " unique-marker"}
    )
    self.assertNotIn(" unique-marker", msgs[0]["content"])

  def test_literal_special_tokens_in_content_pass_through(self):
    """Task content with literal `<|im_end|>` etc. is passed verbatim."""
    weird = "Explain what <|im_end|> means in a chat template."
    model, patcher = self._model_and_patcher()
    req = _FakeRequest(
        context=weird, continuation=" It's the end-of-turn marker."
    )
    with patcher:
      model.loglikelihood([req], disable_tqdm=True)

    msgs = self.captured[0]["json"]["messages"]
    self.assertEqual(msgs[0]["content"], weird)

  def test_score_url_is_v1_chat_score(self):
    """The endpoint URL is the canonical `/v1/chat/score` path."""
    model, patcher = self._model_and_patcher()
    req = _FakeRequest(context="x", continuation=" y")
    with patcher:
      model.loglikelihood([req], disable_tqdm=True)
    self.assertEqual(
        self.captured[0]["url"], "http://127.0.0.1:8080/v1/chat/score"
    )


if __name__ == "__main__":
  absltest.main()
