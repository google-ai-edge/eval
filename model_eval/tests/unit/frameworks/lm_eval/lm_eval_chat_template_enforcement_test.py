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

"""`apply_chat_template=True` contract enforcement test for the LM Eval framework adapter.

Explicitly asserts the operational requirement that `apply_chat_template` is set
to `True` when targeting the LiteRT-LM runner, preventing regressions that would
flatten message lists into text strings prior to reaching the server-side
tokenizer.
"""

from unittest import mock

from absl.testing import absltest

import pytest

pytest.importorskip(
    "lm_eval",
    reason="lm_eval framework adapter requires the package",
)

 # pylint: disable=g-import-not-at-top,g-bad-import-order
from model_eval.frameworks.lm_eval import lm_eval


class LmEvalApplyChatTemplateEnforcementTest(absltest.TestCase):

  def test_apply_chat_template_false_raises_value_error(self):
    fw = lm_eval.LmEvalFramework()
    runner = mock.MagicMock()
    runner.model_name = "test"
    runner.server_url = "http://test"
    runner.returns_greedy = True

    with self.assertRaisesRegex(
        ValueError, r"apply_chat_template must be True"
    ):
      fw.evaluate(
          runner,
          tasks=["arc_easy"],
          limit=1,
          batch_size=1,
          eval_args={"apply_chat_template": False},
      )

  def test_apply_chat_template_true_is_default(self):
    """Default is True (no enforcement raise) — eval proceeds past the check.

    We don't actually run a real eval here (would require
    lm_eval.simple_evaluate
    to be invokable on a fake runner). We just confirm the enforcement
    branch does NOT trigger when the default is in effect.
    """
    fw = lm_eval.LmEvalFramework()
    runner = mock.MagicMock()
    runner.model_name = "test"
    runner.server_url = "http://test"
    runner.returns_greedy = True

    # We patch lm_eval.simple_evaluate to a no-op so the function returns
    # after the enforcement check.
    with mock.patch(
        "model_eval.frameworks.lm_eval.lm_eval.lm_eval.simple_evaluate",
        return_value={"results": {}, "samples": {}},
    ):
      try:
        fw.evaluate(
            runner,
            tasks=["arc_easy"],
            limit=1,
            batch_size=1,
            eval_args={},  # apply_chat_template defaults to True
        )
      except ValueError as e:
        if "apply_chat_template" in str(e):
          self.fail(
              "apply_chat_template enforcement fired with the default True. "
              "The default should NOT raise."
          )
        # other ValueErrors are fine — we only guard the enforcement-specific
        # message.


if __name__ == "__main__":
  absltest.main()
