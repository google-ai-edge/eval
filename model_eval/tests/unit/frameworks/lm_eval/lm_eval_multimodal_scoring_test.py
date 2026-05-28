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

"""Multimodal scoring request rejection verification for `LocalChatScoreModel.loglikelihood`.

Validates that invoking `loglikelihood` with payloads containing multimodal
visual data in `args` triggers an explicit `ValueError`, strictly preventing
silent dropping of images and ensuring failure visibility since the
`/v1/chat/score` endpoint supports text-only requests.
"""

from unittest import mock
from absl.testing import absltest

import pytest

pytest.importorskip(
    "lm_eval",
    reason="lm_eval framework adapter requires the package",
)

  # pylint: disable=g-import-not-at-top,g-bad-import-order
from model_eval.frameworks import _local_chat_score_model


class _FakeRequestWithMultimodal:
  """Mimics lm_eval's request shape for multimodal scoring.

  Per `api_models.py:706-725`, multimodal requests have
  `args = (context, gen_kwargs, auxiliary_args)` where
  `auxiliary_args` carries the visual content. The detector at line
  53 of `_local_chat_score_model.py` is `len(requests[0].args) > 2`.
  """

  def __init__(self):
    self.args = ("context", " continuation", {"visual": "image-data"})


class LmEvalMultimodalScoringRejectionTest(absltest.TestCase):

  def test_multimodal_scoring_raises_value_error(self):
    """A multimodal request (len(args) > 2) MUST raise, not be silently downgraded."""
    model = _local_chat_score_model.LocalChatScoreModel.__new__(
        _local_chat_score_model.LocalChatScoreModel
    )
    model._scoring_url = "http://test/v1/chat/score"
    model.model = "test"

    with self.assertRaisesRegex(
        ValueError, r"[Mm]ultimodal scoring is not supported"
    ):
      model.loglikelihood([_FakeRequestWithMultimodal()], disable_tqdm=True)

  def test_text_only_request_does_not_raise(self):
    """Sanity check: a text-only request (len(args) == 2) passes the multimodal guard."""

    model = _local_chat_score_model.LocalChatScoreModel.__new__(
        _local_chat_score_model.LocalChatScoreModel)
    model._scoring_url = "http://test/v1/chat/score"
    model.model = "test"

    class _TextOnlyReq:

      def __init__(self):
        self.args = ("context", " continuation")

    mock_resp = mock.MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"score": -1.0, "logprobs": {"is_greedy": False}}]
    }
    mock_resp.raise_for_status.return_value = None

    with mock.patch(
        "model_eval.frameworks._local_chat_score_model.requests_lib.post",
        return_value=mock_resp,
    ):
      # Should not raise.
      results = model.loglikelihood([_TextOnlyReq()], disable_tqdm=True)
      self.assertLen(results, 1)


if __name__ == "__main__":
  absltest.main()
