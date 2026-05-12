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

"""Unit tests for _local_chat_score_model.py."""

import unittest
from unittest import mock

from ai_edge_eval.frameworks import _local_chat_score_model


class TestLocalChatScoreModel(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.base_url = "http://127.0.0.1:8080"
    self.model_name = "test-model"
    self.model = _local_chat_score_model.LocalChatScoreModel(
        base_url=self.base_url, model=self.model_name
    )

  @mock.patch(
      "ai_edge_eval.frameworks._local_chat_score_model.requests_lib"
  )
  def test_loglikelihood(self, mock_requests):
    mock_response = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"score": -5.5, "logprobs": {"is_greedy": True}}]
    }
    mock_requests.post.return_value = mock_response

    class MockReq:

      def __init__(self, context, continuation):
        self.args = (context, continuation)

    requests = [
        MockReq("context string", "continuation string"),
        MockReq("", "only continuation"),
    ]

    results = self.model.loglikelihood(requests)

    self.assertEqual(len(results), 2)
    self.assertEqual(results[0], (-5.5, True))
    self.assertEqual(results[1], (-5.5, True))

    self.assertEqual(mock_requests.post.call_count, 2)

    # First call with context.
    expected_payload_1 = {
        "model": self.model_name,
        "messages": [
            {"role": "user", "content": "context string"},
            {"role": "assistant", "content": "continuation string"},
        ],
    }
    mock_requests.post.assert_any_call(
        f"{self.base_url}/v1/chat/score", json=expected_payload_1
    )

    # Second call without context.
    expected_payload_2 = {
        "model": self.model_name,
        "messages": [
            {"role": "assistant", "content": "only continuation"},
        ],
    }
    mock_requests.post.assert_any_call(
        f"{self.base_url}/v1/chat/score", json=expected_payload_2
    )


if __name__ == "__main__":
  unittest.main()
