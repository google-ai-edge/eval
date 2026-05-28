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

"""Unit tests for chat score lighteval model."""

from unittest import mock
from absl.testing import absltest
import pytest

pytest.importorskip(
    "lighteval",
    reason='install via `pip install -e ".[lighteval]"` to run lighteval tests',
)

from model_eval.frameworks.lighteval import _chat_score_lighteval_model  # pylint: disable=g-import-not-at-top,g-bad-import-order


class TestChatScoreLightevalModel(absltest.TestCase):

  def test_chat_score_model(self):
    mock_client = mock.MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_resp = mock.MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"score": -1.5, "logprobs": {"is_greedy": True}}]
    }
    mock_client.post.return_value = mock_resp

    config = mock.MagicMock()
    config.base_url = "http://127.0.0.1:8000/"
    config.model_name = "test-model"

    with mock.patch.object(
        _chat_score_lighteval_model.httpx, "Client"
    ) as mock_httpx_cls, mock.patch(
        "lighteval.models.endpoints.litellm_model.SampleCache"
    ):
      mock_httpx_cls.return_value = mock_client
      model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)

      mock_doc = mock.MagicMock()
      mock_doc.query = "hello"
      mock_doc.choices = ["a", "b", " world"]
      mock_doc.gold_index = 2

      res_ll = model.loglikelihood([mock_doc])
    self.assertLen(res_ll, 1)
    self.assertEqual(res_ll[0].logprobs, [-1.5, -1.5, -1.5])
    self.assertEqual(res_ll[0].argmax_logits_eq_gold, [True, True, True])
    self.assertEqual(mock_client.post.call_count, 3)

    model.loglikelihood_rolling = mock.MagicMock(
        side_effect=NotImplementedError
    )
    with self.assertRaises(NotImplementedError):
      model.loglikelihood_rolling([mock_doc])


if __name__ == "__main__":
  absltest.main()
