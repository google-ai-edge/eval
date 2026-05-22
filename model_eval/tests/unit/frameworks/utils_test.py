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


if __name__ == "__main__":
  unittest.main()
