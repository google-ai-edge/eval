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

"""Unit tests for evaluation framework utils."""

import json
import math
import unittest
from unittest import mock

from model_eval.frameworks import utils


class UtilsTest(unittest.TestCase):

  def test_build_chat_score_messages_with_prompt_attr(self):
    """Tests context containing a JSON prompt history."""
    context = mock.MagicMock()
    context.prompt = json.dumps([{"role": "user", "content": "hello"}])

    messages = utils.build_chat_score_messages(context, "world")

    self.assertEqual(
        messages,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
    )

  def test_build_chat_score_messages_empty_context(self):
    """Tests empty context string or None."""
    messages = utils.build_chat_score_messages("", "world")
    self.assertEqual(messages, [{"role": "assistant", "content": "world"}])

    messages_none = utils.build_chat_score_messages(None, "world")
    self.assertEqual(messages_none, [{"role": "assistant", "content": "world"}])

  def test_build_chat_score_messages_plain_text(self):
    """Tests context as a plain string."""
    messages = utils.build_chat_score_messages("hello", "world")
    self.assertEqual(
        messages,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
    )

  def test_parse_chat_score_response_success(self):
    """Tests successful extraction of score and greedy flag."""
    data = {"choices": [{"score": -1.25, "logprobs": {"is_greedy": True}}]}
    score, is_greedy = utils.parse_chat_score_response(data)

    self.assertEqual(score, -1.25)
    self.assertTrue(is_greedy)

  def test_parse_chat_score_response_missing_or_invalid_choices(self):
    """Tests API response missing the choices array."""
    with self.assertRaisesRegex(ValueError, "choices"):
      utils.parse_chat_score_response({})

    with self.assertRaisesRegex(ValueError, "choices"):
      utils.parse_chat_score_response({"choices": []})

    with self.assertRaisesRegex(ValueError, "choices"):
      utils.parse_chat_score_response({"choices": "not a list"})

  def test_parse_chat_score_response_missing_score(self):
    """Tests API response missing the score key."""
    data = {"choices": [{"logprobs": {"is_greedy": True}}]}
    with self.assertRaisesRegex(ValueError, "score"):
      utils.parse_chat_score_response(data)

  def test_parse_chat_score_response_missing_logprobs(self):
    """Tests API response missing the logprobs or is_greedy keys."""
    with self.assertRaisesRegex(ValueError, "logprobs"):
      utils.parse_chat_score_response({"choices": [{"score": -1.25}]})

    with self.assertRaisesRegex(ValueError, "is_greedy"):
      utils.parse_chat_score_response(
          {"choices": [{"score": -1.25, "logprobs": {}}]}
      )


class BuildChatScoreMessagesEdgeCasesTest(unittest.TestCase):
  """Edge cases for `build_chat_score_messages` not covered by the happy-path tests."""

  def test_empty_continuation_still_produces_assistant_turn(self):
    """An empty continuation must still appear as an assistant turn — that is what gets scored."""
    messages = utils.build_chat_score_messages("Q?", "")
    self.assertEqual(
        messages,
        [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": ""},
        ],
    )

  def test_multi_turn_jsonchatstr_preserved_in_order(self):
    """Multi-turn chat history is unpacked in original order, continuation appended last."""
    history = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "2+2?"},
    ]
    context = mock.MagicMock()
    context.prompt = json.dumps(history)

    messages = utils.build_chat_score_messages(context, "4")

    self.assertEqual(
        messages, history + [{"role": "assistant", "content": "4"}]
    )

  def test_multiline_context_preserved(self):
    """Newlines, leading whitespace, and the typical `Answer:` cue survive untouched."""
    context = (
        "Question: What is the capital of France?\n"
        "Options:\n"
        "  A) Paris\n"
        "  B) London\n"
        "Answer:"
    )
    messages = utils.build_chat_score_messages(context, " A")
    self.assertEqual(messages[0]["content"], context)
    self.assertEqual(messages[1]["content"], " A")

  def test_unicode_and_special_chars_preserved(self):
    """Unicode and shell/JSON-special characters in the continuation are preserved verbatim."""
    weird = "答案：«π ≈ 3.14»\t\"quoted'\\stuff"
    messages = utils.build_chat_score_messages("Q?", weird)
    self.assertEqual(messages[-1]["content"], weird)

  def test_call_is_idempotent_across_invocations(self):
    """Two calls with the same inputs produce structurally equal but independent lists."""
    a = utils.build_chat_score_messages("hi", "yo")
    b = utils.build_chat_score_messages("hi", "yo")
    self.assertEqual(a, b)
    # Mutating one must not bleed into the other.
    a.append({"role": "user", "content": "leaked"})
    self.assertNotEqual(a, b)

  def test_jsonchatstr_with_zero_history_messages(self):
    """A JsonChatStr whose prompt is an empty list yields only the assistant continuation."""
    context = mock.MagicMock()
    context.prompt = json.dumps([])
    messages = utils.build_chat_score_messages(context, "only-assistant")
    self.assertEqual(
        messages, [{"role": "assistant", "content": "only-assistant"}]
    )

  def test_falsy_zero_int_context_treated_as_empty(self):
    """A falsy non-string context (e.g., integer 0) is treated as empty per current contract.

    Documents current behavior so a future change that starts coercing ints to
    strings would break this test loud rather than silently shift semantics.
    """
    messages = utils.build_chat_score_messages(0, "w")
    self.assertEqual(messages, [{"role": "assistant", "content": "w"}])


class ParseChatScoreResponseEdgeCasesTest(unittest.TestCase):
  """Edge cases for `parse_chat_score_response` not covered by the happy-path tests."""

  def test_score_zero_round_trips(self):
    """A score of exactly 0 (deterministic completion under the model) is returned as 0.0."""
    score, is_greedy = utils.parse_chat_score_response(
        {"choices": [{"score": 0, "logprobs": {"is_greedy": True}}]}
    )
    self.assertEqual(score, 0.0)
    self.assertIs(type(score), float)  # not int
    self.assertTrue(is_greedy)

  def test_score_negative_infinity_returns_float_inf(self):
    """An impossible continuation (logprob = -inf) round-trips as math.isinf(-...)."""
    data = {
        "choices": [{
            "score": float("-inf"),
            "logprobs": {"is_greedy": False},
        }]
    }
    score, _ = utils.parse_chat_score_response(data)
    self.assertTrue(math.isinf(score) and score < 0)

  def test_int_score_coerced_to_float(self):
    """A server that returns score as a Python `int` (e.g., -1) is coerced to float."""
    score, _ = utils.parse_chat_score_response(
        {"choices": [{"score": -1, "logprobs": {"is_greedy": False}}]}
    )
    self.assertEqual(score, -1.0)
    self.assertIs(type(score), float)

  def test_is_greedy_truthy_int_coerced_to_bool(self):
    """An `is_greedy` field given as 1/0 is coerced to True/False."""
    score, ig_true = utils.parse_chat_score_response(
        {"choices": [{"score": -0.1, "logprobs": {"is_greedy": 1}}]}
    )
    self.assertEqual(score, -0.1)
    self.assertIs(ig_true, True)

    _, ig_false = utils.parse_chat_score_response(
        {"choices": [{"score": -0.1, "logprobs": {"is_greedy": 0}}]}
    )
    self.assertIs(ig_false, False)

  def test_logprobs_not_a_dict_raises(self):
    """`logprobs` field that isn't a dict (e.g., None, a list, a string) raises ValueError."""
    for bad in (None, [], "not-a-dict", 42):
      with self.assertRaisesRegex(ValueError, "logprobs"):
        utils.parse_chat_score_response(
            {"choices": [{"score": -0.1, "logprobs": bad}]}
        )

  def test_only_first_choice_is_used(self):
    """When multiple choices are returned, only the first one's score is read."""
    data = {
        "choices": [
            {"score": -1.0, "logprobs": {"is_greedy": True}},
            {"score": -99.0, "logprobs": {"is_greedy": False}},
        ]
    }
    score, is_greedy = utils.parse_chat_score_response(data)
    self.assertEqual(score, -1.0)
    self.assertTrue(is_greedy)

  def test_extra_unknown_fields_are_ignored(self):
    """Forward-compatible: unrecognized top-level and choice-level fields don't break parsing."""
    data = {
        "object": "chat.score",
        "model": "some-model",
        "usage": {"prompt_tokens": 10},  # extra
        "choices": [{
            "index": 0,
            "score": -2.5,
            "logprobs": {
                "is_greedy": False,
                "token_logprobs": [-0.1, -2.4],  # extra
            },
            "finish_reason": "stop",  # extra
        }],
    }
    score, is_greedy = utils.parse_chat_score_response(data)
    self.assertEqual(score, -2.5)
    self.assertFalse(is_greedy)


if __name__ == "__main__":
  unittest.main()
