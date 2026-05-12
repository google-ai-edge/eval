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

"""Unit tests for lm-eval adapter."""

import unittest
from unittest import mock

from ai_edge_eval.api import constants
from ai_edge_eval.frameworks import base
from ai_edge_eval.frameworks import lm_eval
from ai_edge_eval.runners import base as runners_base


class TestLmEvalAdapter(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_runner = mock.MagicMock(spec=runners_base.AbstractRunner)
    self.mock_runner.server_url = "http://127.0.0.1:8080"
    self.mock_runner.model_name = "test-model"
    # pylint: disable=protected-access
    self.mock_runner._config = mock.MagicMock()
    self.mock_runner._config.tokenizer_path = "/path/to/tokenizer"
    self.mock_runner.__enter__.return_value = self.mock_runner
    # pylint: enable=protected-access

  @mock.patch("ai_edge_eval.frameworks.lm_eval.lm_eval")
  @mock.patch("ai_edge_eval.frameworks.lm_eval.lm_eval_tasks")
  def test_evaluate(self, mock_tasks, mock_lm_eval):

    mock_lm_eval.simple_evaluate.return_value = {
        "results": {"task1": {"acc": 0.9}},
        "samples": {"task1": []},
        "versions": {"task1": "1.0"},
        "n-shot": {"task1": 0},
        "higher_is_better": {"task1": True},
        "config": {"model": "local-chat-completions"},
        "lm_eval_version": "0.4",
    }

    framework = lm_eval.LmEvalFramework()
    results = framework.evaluate(
        self.mock_runner, ["task1"], eval_args={"apply_chat_template": True}
    )

    self.mock_runner.__enter__.assert_called_once()
    mock_lm_eval.simple_evaluate.assert_called_once_with(
        model=constants.LOCAL_CHAT_SCORE_MODEL_NAME,
        model_args="base_url=http://127.0.0.1:8080,model=test-model",
        tasks=["task1"],
        num_fewshot=0,
        batch_size=1,
        limit=None,
        apply_chat_template=True,
        task_manager=mock.ANY,
    )
    self.assertEqual(results.framework_type, "lm-eval")
    self.assertEqual(results.aggregated_metrics["task1"]["acc"], 0.9)
    self.assertEqual(results.metadata["lm_eval_version"], "0.4")

  @mock.patch("ai_edge_eval.frameworks.lm_eval.lm_eval")
  @mock.patch("ai_edge_eval.frameworks.lm_eval.lm_eval_tasks")
  def test_evaluate_native(self, mock_tasks, mock_lm_eval):

    mock_lm_eval.simple_evaluate.return_value = {
        "results": {"task2": {"acc": 0.95}},
        "samples": {"task2": []},
        "versions": {"task2": "1.0"},
        "n-shot": {"task2": 5},
        "higher_is_better": {"task2": True},
        "config": {"model": "hf"},
        "lm_eval_version": "0.4",
    }

    framework = lm_eval.LmEvalFramework()
    config = base.NativeModelConfig(
        model="hf", model_args={"pretrained": "gpt2"}
    )
    results = framework.evaluate_native(config, ["task2"])

    mock_lm_eval.simple_evaluate.assert_called_once_with(
        model="hf",
        model_args={"pretrained": "gpt2"},
        tasks=["task2"],
        num_fewshot=0,
        batch_size=None,
        device=None,
        limit=None,
        apply_chat_template=True,
        task_manager=mock.ANY,
    )
    self.assertEqual(results.framework_type, "lm-eval")
    self.assertEqual(results.aggregated_metrics["task2"]["acc"], 0.95)

  @mock.patch("ai_edge_eval.frameworks.lm_eval.lm_eval")
  @mock.patch("ai_edge_eval.frameworks.lm_eval.lm_eval_tasks")
  def test_evaluate_warns_loglikelihood_tasks(self, mock_tasks, mock_lm_eval):
    mock_task = mock.MagicMock()
    mock_task.OUTPUT_TYPE = "loglikelihood"
    mock_tasks.TaskManager.return_value.load_task_or_group.return_value = {
        "task3": mock_task
    }

    self.mock_runner.returns_greedy = False
    framework = lm_eval.LmEvalFramework()

    with mock.patch("warnings.warn") as mock_warn:
      framework.evaluate(
          self.mock_runner, ["task3"], eval_args={"apply_chat_template": True}
      )
      mock_warn.assert_called_once()

  def test_evaluate_conflicts(self):
    framework = lm_eval.LmEvalFramework()
    with self.assertRaisesRegex(
        ValueError, "--limit conflicts with 'limit' in --eval-args"
    ):
      framework.evaluate(
          self.mock_runner,
          ["task1"],
          limit=10,
          eval_args={"limit": 10, "apply_chat_template": True},
      )

    with self.assertRaisesRegex(
        ValueError, "--batch-size conflicts with 'batch_size' in --eval-args"
    ):
      framework.evaluate(
          self.mock_runner,
          ["task1"],
          batch_size=2,
          eval_args={"batch_size": 2, "apply_chat_template": True},
      )

  def test_evaluate_requires_apply_chat_template(self):
    framework = lm_eval.LmEvalFramework()
    with self.assertRaisesRegex(ValueError, "apply_chat_template must be True"):
      framework.evaluate(
          self.mock_runner, ["task1"], eval_args={"apply_chat_template": False}
      )

  def test_evaluate_native_conflicts(self):
    framework = lm_eval.LmEvalFramework()
    config = base.NativeModelConfig(
        model="hf",
        model_path="/foo/bar",
        model_args={"pretrained": "/other/path"},
    )
    with self.assertRaisesRegex(
        ValueError, "--model-path conflicts with 'pretrained' in --runner-args"
    ):
      framework.evaluate_native(config, ["task2"])

    config_device = base.NativeModelConfig(
        model="hf",
        device="cuda",
        model_args={"device": "cpu"},
    )
    with self.assertRaisesRegex(
        ValueError, "--device conflicts with 'device' in --runner-args"
    ):
      framework.evaluate_native(config_device, ["task2"])

  @mock.patch("ai_edge_eval.frameworks.lm_eval.lm_eval")
  @mock.patch("ai_edge_eval.frameworks.lm_eval.lm_eval_tasks")
  def test_evaluate_native_with_top_level_attributes(
      self, mock_tasks, mock_lm_eval
  ):
    mock_lm_eval.simple_evaluate.return_value = {}
    framework = lm_eval.LmEvalFramework()
    config = base.NativeModelConfig(
        model="hf",
        model_path="/model/weights",
        device="npu",
    )
    framework.evaluate_native(config, ["task2"])
    mock_lm_eval.simple_evaluate.assert_called_once_with(
        model="hf",
        model_args={"pretrained": "/model/weights"},
        tasks=["task2"],
        num_fewshot=0,
        batch_size=None,
        device="npu",
        limit=None,
        apply_chat_template=True,
        task_manager=mock.ANY,
    )

  def test_describe_eval_args(self):
    fields = lm_eval.LmEvalFramework.describe_eval_args()
    self.assertIsInstance(fields, list)
    for field in fields:
      self.assertIn("name", field)

  def test_describe_native_runner_args(self):
    fields = lm_eval.LmEvalFramework.describe_native_runner_args("hf")
    self.assertIsInstance(fields, list)
    for field in fields:
      self.assertIn("name", field)

  def test_describe_native_runner_args_unknown(self):
    with self.assertRaises(NotImplementedError):
      lm_eval.LmEvalFramework.describe_native_runner_args("non_existent_runner")

  @mock.patch("inspect.signature")
  @mock.patch("lm_eval.api.registry")
  @mock.patch("ai_edge_eval.frameworks.lm_eval.lm_eval")
  def test_evaluate_native_dynamic_path_key(
      self, mock_lm_eval, mock_registry, mock_sig
  ):
    framework = lm_eval.LmEvalFramework()
    config = base.NativeModelConfig(
        model="test_model", model_path="/path/to/weights"
    )

    # Mocking a model signature with 'pretrained'
    mock_params = mock.MagicMock()
    mock_params.parameters = {"pretrained": mock.Mock()}
    mock_sig.return_value = mock_params

    # Setup for from_unified_runner_args (which calls _consume_arg)
    framework.evaluate_native(config, ["task1"])

    # Verify simple_evaluate was called with 'pretrained'
    mock_lm_eval.simple_evaluate.assert_called_once()
    actual_model_args = mock_lm_eval.simple_evaluate.call_args[1]["model_args"]
    self.assertEqual(actual_model_args["pretrained"], "/path/to/weights")

  @mock.patch("ai_edge_eval.runners.registry.get_all_runners")
  def test_supported_runners(self, mock_get_all_runners):
    mock_get_all_runners.return_value = ["custom_b", "custom_a"]

    with mock.patch(
        "lm_eval.api.registry.MODEL_REGISTRY",
        {"native_c": None, "custom_a": None},
    ):
      runners = lm_eval.LmEvalFramework.supported_runners()
      self.assertEqual(runners, ["custom_a", "custom_b", "native_c"])


if __name__ == "__main__":
  unittest.main()
