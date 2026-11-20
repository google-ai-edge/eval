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
from lighteval.logging import info_loggers
from lighteval.models import model_output
from lighteval.tasks import requests as lighteval_requests


class TestLightEvalAdapter(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_runner = mock.MagicMock(spec=runner_base.AbstractRunner)
    self.mock_runner.server_url = "http://127.0.0.1:8080"
    self.mock_runner.server_args = {"model_name": "test_model"}
    self.mock_runner.request_timeout_sec = None

  @mock.patch.object(lighteval.lighteval_pipeline, "Pipeline")
  def test_evaluate(self, mock_pipeline_cls):
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.get_results.return_value = {
        "results": {"task1": {"acc": 0.9}},
        "samples": {},
        "config": {"model": "test_model"},
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
    self.assertEqual(kwargs["model"]._model_name, "openai/test_model")
    self.assertEqual(kwargs["model"]._timeout_sec, 120.0)

    mock_pipeline.evaluate.assert_called_once()
    mock_pipeline.save_and_push_results.assert_called_once()

    self.assertEqual(results.framework_type, "lighteval")
    self.assertEqual(results.aggregated_metrics["task1"]["acc"], 0.9)

  @mock.patch.object(lighteval.lighteval_pipeline, "Pipeline")
  def test_evaluate_defaults_model_name_and_forwards_timeout(
      self, mock_pipeline_cls
  ):
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.get_results.return_value = {"results": {}, "samples": {}}
    mock_pipeline.evaluation_tracker.details = {}
    self.mock_runner.server_args = {}
    self.mock_runner.request_timeout_sec = 45.0
    lighteval.LightEvalFramework().evaluate(self.mock_runner, ["task1"])
    model = mock_pipeline_cls.call_args.kwargs["model"]
    self.assertEqual(model._model_name, "openai/default_model")
    self.assertEqual(model._timeout_sec, 45.0)

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

  @mock.patch.object(lighteval.lighteval_pipeline, "Pipeline")
  def test_evaluate_native_switches_to_vlm_config_when_vision_model_true(
      self, mock_pipeline_cls
  ):
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.get_results.return_value = {
        "results": {"task_vlm": {"acc": 0.88}},
        "samples": {},
        "config": {"model": "accelerate"},
    }
    mock_pipeline.evaluation_tracker.details = {}

    framework = lighteval.LightEvalFramework()
    config = base.NativeModelConfig(
        model="accelerate",
        model_args={"model_name": "dummy_vlm", "vision_model": True},
    )
    with mock.patch.object(lighteval.importlib, "import_module") as mock_import:
      mock_vlm_cls = mock.MagicMock()

      def fake_import(path):
        if path == "lighteval.models.transformers.vlm_transformers_model":
          m = mock.MagicMock()
          m.VLMTransformersModelConfig = mock_vlm_cls
          return m
        return mock.MagicMock()

      mock_import.side_effect = fake_import

      results = framework.evaluate_native(config, ["task_vlm"])

    mock_vlm_cls.assert_called_once()
    self.assertNotIn("vision_model", mock_vlm_cls.call_args.kwargs)
    self.assertEqual(results.framework_type, "lighteval")
    self.assertEqual(results.aggregated_metrics["task_vlm"]["acc"], 0.88)

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

  def test_subtasks_of_expands_suite(self):
    """Verifies that suite names resolve to colon-separated leaf tasks."""
    mmlu_subs = lighteval.LightEvalFramework.subtasks_of("mmlu")
    self.assertIn("mmlu:abstract_algebra", mmlu_subs)
    # Ensure the result is a non-trivial list of colon-prefixed leaves
    # rather than groups, so unioning with the parent gives a real
    # expansion of the allowlist.
    self.assertTrue(all(":" in s for s in mmlu_subs))
    self.assertGreater(len(mmlu_subs), 1)

  def test_subtasks_of_unknown_returns_empty(self):
    """Ensures unrecognized tasks or leaf tasks produce an empty expansion list."""
    self.assertEqual(
        lighteval.LightEvalFramework.subtasks_of("not_a_suite"), []
    )
    # A concrete leaf is not itself a suite, so it should also return [].
    self.assertEqual(
        lighteval.LightEvalFramework.subtasks_of("mmlu:abstract_algebra"), []
    )


class TestLightEvalDetailsLogger(absltest.TestCase):

  def test_log_and_aggregate_hashes_utf8(self):
    """Regression test for xxhash >= 4.0 rejecting unencoded str inputs."""
    logger = info_loggers.DetailsLogger()
    logger.log(
        task_name="task_a",
        doc=lighteval_requests.Doc(
            query="What is the capital of France?",
            choices=["Paris", "Lyon"],
            gold_index=0,
        ),
        model_response=model_output.ModelResponse(
            text=["Paris"],
            input_tokens=[1, 2, 3],
            output_tokens=[[4, 5]],
        ),
        metrics={"acc": 1.0},
    )
    logger.log(
        task_name="task_b",
        doc=lighteval_requests.Doc(
            query="日本の首都はどこですか？",
            choices=["東京", "大阪"],
            gold_index=0,
        ),
        model_response=model_output.ModelResponse(
            text=["東京"],
            input_tokens=[10, 20],
            output_tokens=[[30]],
        ),
        metrics={"acc": 1.0},
    )

    logger.aggregate()

    expected_hash_keys = (
        "hash_examples",
        "hash_full_prompts",
        "hash_input_tokens",
        "hash_cont_tokens",
    )
    hex64_pattern = r"^[0-9a-f]{16}$"

    for task_name in ("task_a", "task_b"):
      sample_hash = logger.hashes[task_name][0]
      self.assertRegex(sample_hash.example, hex64_pattern)
      self.assertRegex(sample_hash.input_tokens, hex64_pattern)
      self.assertRegex(sample_hash.cont_tokens, hex64_pattern)

      compiled = logger.compiled_details[task_name].hashes
      for key in expected_hash_keys:
        self.assertRegex(compiled[key], hex64_pattern)

    self.assertNotEqual(
        logger.hashes["task_a"][0].example,
        logger.hashes["task_b"][0].example,
    )

    overall = logger.compiled_details_over_all_tasks.hashes
    for key in expected_hash_keys:
      self.assertRegex(overall[key], hex64_pattern)


if __name__ == "__main__":
  absltest.main()
