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

"""Unit tests for lighteval adapter."""

from unittest import mock
from absl.testing import absltest
import pytest

pytest.importorskip(
    "lighteval",
    reason='install via `pip install -e ".[lighteval]"` to run lighteval tests',
)

# pylint: disable=g-import-not-at-top,g-bad-import-order
from model_eval.frameworks import base
from model_eval.frameworks.lighteval import lighteval
from model_eval.runners import base as runner_base


class TestLightEvalAdapter(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_runner = mock.MagicMock(spec=runner_base.AbstractRunner)
    self.mock_runner.server_url = "http://127.0.0.1:8080"
    self.mock_runner.model_name = "test-model"

  @mock.patch.object(lighteval.lighteval_pipeline, "Pipeline")
  def test_evaluate(self, mock_pipeline_cls):
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.get_results.return_value = {
        "results": {"task1": {"acc": 0.9}},
        "samples": {},
        "config": {"model": "test-model"},
    }
    mock_pipeline.evaluation_tracker.details = {}

    framework = lighteval.LightEvalFramework()
    results = framework.evaluate(
        self.mock_runner, ["task1"], sample_range=(0, 99)
    )

    mock_pipeline_cls.assert_called_once()
    kwargs = mock_pipeline_cls.call_args.kwargs
    self.assertEqual(kwargs["tasks"], "task1")
    self.assertEqual(kwargs["pipeline_parameters"].max_samples, 100)
    self.assertIsNotNone(kwargs["model"])

    mock_pipeline.evaluate.assert_called_once()
    mock_pipeline.save_and_push_results.assert_called_once()

    self.assertEqual(results.framework_type, "lighteval")
    self.assertEqual(results.aggregated_metrics["task1"]["acc"], 0.9)

  @mock.patch.object(lighteval.lighteval_pipeline, "Pipeline")
  def test_evaluate_native(self, mock_pipeline_cls):
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.get_results.return_value = {
        "results": {"task2": {"acc": 0.95}},
        "samples": {},
        "config": {"model": "accelerate"},
    }
    mock_pipeline.evaluation_tracker.details = {}

    framework = lighteval.LightEvalFramework()
    config = base.NativeModelConfig(
        model="accelerate", model_args={"model_name": "gpt2"}
    )
    with mock.patch.object(lighteval.importlib, "import_module") as mock_import:
      mock_import.return_value = mock.MagicMock()
      results = framework.evaluate_native(
          config, ["task2"], sample_range=(0, 49)
      )

    mock_pipeline.evaluate.assert_called_once()
    mock_pipeline.save_and_push_results.assert_called_once()

    self.assertEqual(results.framework_type, "lighteval")
    self.assertEqual(results.aggregated_metrics["task2"]["acc"], 0.95)

  def test_evaluate_native_conflicts(self):
    framework = lighteval.LightEvalFramework()
    with mock.patch.object(lighteval.importlib, "import_module") as mock_import:
      mock_import.return_value = mock.MagicMock()
      config = base.NativeModelConfig(
          model="accelerate",
          model_path="path/to/model",
          model_args={"model_name": "other/path"},
      )
      with self.assertRaisesRegex(
          ValueError,
          "--model-path conflicts with 'model_name' in --runner-args",
      ):
        framework.evaluate_native(config, ["task2"])

      config_device = base.NativeModelConfig(
          model="accelerate", device="cuda", model_args={"device": "cpu"}
      )
      with self.assertRaisesRegex(
          ValueError, "--device conflicts with 'device' in --runner-args"
      ):
        framework.evaluate_native(config_device, ["task2"])

  def test_conflict_batch_size(self):
    framework = lighteval.LightEvalFramework()
    with self.assertRaisesRegex(
        ValueError, "conflicts with 'batch_size' in --eval-args"
    ):
      framework.evaluate(
          self.mock_runner,
          ["task1"],
          batch_size=8,
          eval_args={"batch_size": 8},
      )

  def test_describe_eval_args(self):
    framework = lighteval.LightEvalFramework()
    fields = framework.describe_eval_args()
    self.assertIsInstance(fields, list)
    for field in fields:
      self.assertIn("name", field)

  def test_describe_native_runner_args(self):
    framework = lighteval.LightEvalFramework()
    with mock.patch.object(
        lighteval.introspection, "get_fields"
    ) as mock_get_fields:
      with mock.patch.object(
          lighteval.importlib, "import_module"
      ) as mock_import_module:
        mock_get_fields.return_value = [{"name": "model_name"}]
        mock_import_module.return_value = mock.MagicMock()
        fields = framework.describe_native_runner_args("accelerate")

    self.assertIsInstance(fields, list)
    for field in fields:
      self.assertIn("name", field)

  @mock.patch("model_eval.runners.registry.get_all_runners")
  def test_supported_runners(self, mock_get_all_runners):
    mock_get_all_runners.return_value = ["custom_a", "custom_b"]
    runners = lighteval.LightEvalFramework.supported_runners()
    self.assertEqual(
        runners, ["accelerate", "custom_a", "custom_b"]
    )


if __name__ == "__main__":
  absltest.main()
