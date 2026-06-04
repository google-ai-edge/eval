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

"""Unit tests for EvalPipeline."""

import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest
import pytest

pytest.importorskip(
    "lighteval",
    reason=(
        'install via `pip install -e ".[lighteval]"` to run pipeline tests'
        " (the pipeline module transitively imports lighteval)"
    ),
)

# pylint: disable=g-import-not-at-top,g-bad-import-order
from model_eval.api import pipeline
from model_eval.frameworks import base as framework_base
from model_eval.runners import base as runner_base
from model_eval.runners import litert_lm
import yaml


class EvalPipelineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.output_dir = self.temp_dir.name

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def test_unknown_task_raises_value_error(self):
    """Verifies that an unknown task raises a ValueError."""
    with self.assertRaisesRegex(ValueError, "Tasks not in allowlist"):
      pipeline.EvalPipeline(
          model="test_model",
          tasks=("unknown_task",),
          framework=framework_base.FrameworkType.LM_EVAL,
          eval_args={"apply_chat_template": True},
          output_dir=self.output_dir,
      )

  @mock.patch("model_eval.api.pipeline._load_runner_allowlist")
  @mock.patch("model_eval.api.pipeline._load_task_allowlist")
  def test_subtask_authorized_by_parent_in_allowlist(
      self, mock_load_task, mock_load_runner
  ):
    """Ensures a subtask is permitted if its parent group is allowlisted."""
    mock_load_task.return_value = ["mmlu"]
    mock_load_runner.return_value = ["native"]
    with mock.patch.object(
        pipeline.framework_registry,
        "get_framework",
    ) as mock_get_framework:
      mock_fw = mock.MagicMock()
      mock_fw.subtasks_of.return_value = ["mmlu_abstract_algebra"]
      mock_get_framework.return_value = mock_fw

      p = pipeline.EvalPipeline(
          model="test_model",
          tasks=("mmlu_abstract_algebra",),
          framework=framework_base.FrameworkType.LM_EVAL,
          eval_args={"apply_chat_template": True},
          output_dir=self.output_dir,
      )
    self.assertEqual(p._tasks, ["mmlu_abstract_algebra"])

  @mock.patch("model_eval.api.pipeline._load_runner_allowlist")
  @mock.patch("model_eval.api.pipeline._load_task_allowlist")
  def test_unrelated_task_still_rejected_with_subtask_expansion(
      self, mock_load_task, mock_load_runner
  ):
    """Confirms that unauthorized tasks remain blocked under subtask expansion."""
    mock_load_task.return_value = ["mmlu"]
    mock_load_runner.return_value = ["native"]
    with mock.patch.object(
        pipeline.framework_registry,
        "get_framework",
    ) as mock_get_framework:
      mock_fw = mock.MagicMock()
      mock_fw.subtasks_of.return_value = ["mmlu_abstract_algebra"]
      mock_get_framework.return_value = mock_fw

      with self.assertRaisesRegex(ValueError, "Tasks not in allowlist"):
        pipeline.EvalPipeline(
            model="test_model",
            tasks=("gsm8k",),
            framework=framework_base.FrameworkType.LM_EVAL,
            eval_args={"apply_chat_template": True},
            output_dir=self.output_dir,
        )

  @mock.patch(
      "model_eval.api.pipeline._load_runner_allowlist"
  )
  @mock.patch("model_eval.api.pipeline._load_task_allowlist")
  def test_runner_resolution_falls_back_to_model(
      self, mock_load_task, mock_load_runner
  ):
    """Verifies that runner resolution falls back to model attribute."""
    mock_load_task.return_value = ["mmlu"]

    native_cfg = framework_base.NativeModelConfig(model="hf")

    # Case 1: Allowlist contains the resolved model name.
    mock_load_runner.return_value = ["hf"]
    pipeline.EvalPipeline(
        model=native_cfg,
        tasks=("mmlu",),
        framework=framework_base.FrameworkType.LM_EVAL,
        output_dir=self.output_dir,
    )

    # Case 2: Allowlist does not contain the resolved model name.
    mock_load_runner.return_value = ["native"]
    with self.assertRaisesRegex(
        ValueError, "Runner not in allowlist for framework"
    ):
      pipeline.EvalPipeline(
          model=native_cfg,
          tasks=("mmlu",),
          framework=framework_base.FrameworkType.LM_EVAL,
          output_dir=self.output_dir,
      )

  @mock.patch(
      "model_eval.api.pipeline._load_runner_allowlist"
  )
  @mock.patch("model_eval.api.pipeline._load_task_allowlist")
  def test_known_task_passes_validation(self, mock_load_task, mock_load_runner):
    """Verifies that a known task passes validation."""
    mock_load_task.return_value = ["mmlu"]
    mock_load_runner.return_value = ["native"]
    p = pipeline.EvalPipeline(
        model="test_model",
        tasks=("mmlu",),
        framework=framework_base.FrameworkType.LM_EVAL,
        eval_args={"apply_chat_template": True},
        output_dir=self.output_dir,
    )
    self.assertEqual(p._tasks, ["mmlu"])

  @mock.patch(
      "model_eval.api.pipeline._load_runner_allowlist"
  )
  def test_custom_task_config(self, mock_load_runner):
    """Verifies that custom task configurations are correctly loaded and validated."""
    mock_load_runner.return_value = ["native"]
    custom_config = {"lm_eval": ["my_custom_task"]}
    config_path = os.path.join(self.output_dir, "custom_tasks.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      yaml.dump(custom_config, f)

    p = pipeline.EvalPipeline(
        model="test_model",
        tasks=("my_custom_task",),
        framework=framework_base.FrameworkType.LM_EVAL,
        eval_args={"apply_chat_template": True},
        output_dir=self.output_dir,
        task_config=config_path,
    )
    self.assertEqual(p._tasks, ["my_custom_task"])

  def test_custom_runner_config_allowlist_validation(self):
    """Verifies that custom runner configurations are correctly applied."""
    custom_config = {"lm_eval": ["my_custom_runner"]}
    config_path = os.path.join(self.output_dir, "custom_runners.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      yaml.dump(custom_config, f)

    with self.assertRaisesRegex(
        ValueError, "Runner not in allowlist for framework"
    ):
      pipeline.EvalPipeline(
          model="test_model",
          tasks=(),
          framework=framework_base.FrameworkType.LM_EVAL,
          output_dir=self.output_dir,
          runner_config=config_path,
      )

  def test_validate_raises_on_unsupported_native_framework(self):
    """Verifies that _validate raises when using a NativeModelConfig on an unsupported framework."""
    native_cfg = framework_base.NativeModelConfig(model="mock-model")
    with self.assertRaisesRegex(
        ValueError, "does not support native model configs"
    ):
      pipeline.EvalPipeline(
          model=native_cfg,
          framework="non-existent-framework",  # pytype: disable=wrong-arg-types
          output_dir=self.output_dir,
      )

  @mock.patch(
      "model_eval.api.pipeline._load_runner_allowlist"
  )
  def test_validate_custom_framework_raises_with_native_config(
      self, mock_load_runner
  ):
    """Verifies that using framework='custom' with a NativeModelConfig raises a ValueError."""
    mock_load_runner.return_value = ["mock-model"]
    native_cfg = framework_base.NativeModelConfig(model="mock-model")
    with self.assertRaisesRegex(
        ValueError, "framework='custom' requires a custom runner config"
    ):
      pipeline.EvalPipeline(
          model=native_cfg,
          framework=framework_base.FrameworkType.CUSTOM,
          output_dir=self.output_dir,
      )

  @mock.patch("model_eval.api.pipeline._load_task_allowlist")
  @mock.patch("model_eval.runners.registry.create_runner")
  @mock.patch("model_eval.frameworks.registry.get_framework")
  def test_run_produces_jsonl_files(
      self, mock_get_framework, mock_create_runner, mock_load
  ):
    """Verifies that executing the pipeline produces metrics and samples JSONL files."""
    mock_load.return_value = ["mmlu"]
    mock_runner = mock.MagicMock()
    mock_create_runner.return_value = mock_runner
    mock_framework = mock.MagicMock()
    mock_get_framework.return_value = mock_framework

    # Mock evaluation results.
    mock_results = framework_base.EvalResults(
        framework_type="lm-eval",
        aggregated_metrics={"mmlu": {"acc": 0.5, "acc_norm": 0.6}},
        per_sample_outputs={
            "mmlu": [{"doc": "test doc", "target": "A", "res": "A"}]
        },
    )
    mock_framework.evaluate.return_value = mock_results

    p = pipeline.EvalPipeline(
        model=litert_lm.LiteRtLmRunner.Config(
            runner_type="litert-lm",
            model_path="test_model",
        ),
        tasks=("mmlu",),
        framework=framework_base.FrameworkType.LM_EVAL,
        eval_args={"apply_chat_template": True},
        output_dir=self.output_dir,
        sample_range=(0, 10),
    )

    returned_results = p.run()
    self.assertEqual(returned_results, mock_results)

    # Check that it called evaluate and create_runner correctly.
    mock_framework.evaluate.assert_called_once()
    mock_create_runner.assert_called_once_with(
        runner_base.RunnerType.LITERT_LM,
        {
            "runner_type": "litert-lm",
            "model_path": "test_model",
            "backend": "cpu",
            "vision_backend": None,
            "audio_backend": None,
            "port": 8080,
            "host": "127.0.0.1",
            "max_num_tokens": 4096,
            "enable_speculative_decoding": None,
            "model_name": "litert-lm-model",
            "min_log_severity": 1000,
            "always_return_not_greedy": True,
        },
    )

    # Check files created
    files = os.listdir(self.output_dir)
    metric_files = [f for f in files if f.endswith("_metrics.jsonl")]
    sample_files = [f for f in files if f.endswith("_samples.jsonl")]

    self.assertEqual(len(metric_files), 1)
    self.assertEqual(len(sample_files), 1)

    with open(os.path.join(self.output_dir, metric_files[0]), "r") as f:
      lines = f.readlines()
      self.assertEqual(len(lines), 2)
      data = json.loads(lines[0])
      self.assertEqual(data["task"], "mmlu")
      self.assertEqual(data["metric"], "acc")
      self.assertEqual(data["value"], 0.5)

    with open(os.path.join(self.output_dir, sample_files[0]), "r") as f:
      lines = f.readlines()
      self.assertEqual(len(lines), 1)
      data = json.loads(lines[0])
      self.assertEqual(data["task"], "mmlu")
      self.assertEqual(data["res"], "A")

  def test_describe_runner_args(self):
    """Verifies that describe_runner_args returns formatted arguments for a known runner."""
    desc = pipeline.EvalPipeline.describe_runner_args("litert-lm")
    self.assertIn("Runner args for: litert-lm", desc)
    self.assertIn("model_path", desc)

  def test_describe_runner_args_unknown(self):
    """Verifies that describe_runner_args returns unknown status for an invalid runner."""
    desc = pipeline.EvalPipeline.describe_runner_args("invalid-runner")
    self.assertEqual(desc, "Unknown runner: 'invalid-runner'")

  def test_describe_eval_args(self):
    """Verifies that describe_eval_args returns formatted evaluation args for a known framework."""
    desc = pipeline.EvalPipeline.describe_eval_args("lm-eval")
    self.assertIn("Eval args for: lm-eval", desc)

  def test_describe_eval_args_unknown(self):
    """Verifies that describe_eval_args returns unknown status for an invalid framework."""
    desc = pipeline.EvalPipeline.describe_eval_args("invalid-framework")
    self.assertEqual(desc, "Unknown framework: 'invalid-framework'")

  def test_format_fields_with_columns(self):
    """Verifies that _format_fields includes column names and a separator."""
    fields = [{"name": "path", "type": "str", "default": "None"}]
    output = pipeline.EvalPipeline._format_fields(fields, "Header")

    lines = output.strip().split("\n")
    self.assertIn("Argument", lines[1])
    self.assertIn("Type", lines[1])
    self.assertIn("Default", lines[1])
    self.assertTrue(all(c == "-" or c == " " for c in lines[2]))
    self.assertIn("path", lines[3])

  def test_format_fields_empty(self):
    """Verifies that _format_fields handles empty input without error."""
    # pylint: disable=protected-access
    output = pipeline.EvalPipeline._format_fields([], "Empty Header")
    self.assertEqual(output.strip(), "Empty Header")

  @mock.patch("model_eval.api.pipeline._load_task_allowlist")
  @mock.patch(
      "model_eval.frameworks.lighteval.lighteval.LightEvalFramework._run_pipeline"
  )
  def test_pipeline_dispatch_lighteval_native(
      self, mock_run_pipeline, mock_load
  ):
    mock_load.return_value = ["example_fallback_task"]
    mock_res = mock.MagicMock()
    mock_res.aggregated_metrics = {}
    mock_res.per_sample_outputs = {}
    mock_run_pipeline.return_value = mock_res

    model = framework_base.NativeModelConfig(
        model="accelerate", model_args={"model_name": "gpt2"}
    )
    pipe = pipeline.EvalPipeline(
        model=model,
        framework="lighteval",
        tasks=("example_fallback_task",),
        output_dir=self.output_dir,
    )
    pipe.run()
    mock_run_pipeline.assert_called_once()

  @mock.patch("model_eval.api.pipeline._load_task_allowlist")
  @mock.patch(
      "model_eval.frameworks.lm_eval.lm_eval.LmEvalFramework.evaluate_native"
  )
  def test_pipeline_dispatch_lm_eval_native(
      self, mock_evaluate_native, mock_load
  ):
    mock_load.return_value = ["example_task"]
    mock_res = mock.MagicMock()
    mock_res.aggregated_metrics = {}
    mock_res.per_sample_outputs = {}
    mock_evaluate_native.return_value = mock_res

    model = framework_base.NativeModelConfig(
        model="hf", model_args={"pretrained": "gpt2"}
    )
    pipe = pipeline.EvalPipeline(
        model=model,
        framework=framework_base.FrameworkType.LM_EVAL,
        tasks=("example_task",),
        output_dir=self.output_dir,
    )
    pipe.run()
    mock_evaluate_native.assert_called_once()

  @mock.patch(
      "model_eval.custom_tasks.registry.TaskRegistry.global_registry"
  )
  @mock.patch(
      "model_eval.api.pipeline._load_runner_allowlist"
  )
  def test_custom_framework_tasks_allowlist(
      self, mock_load_runner, mock_global_registry
  ):
    mock_load_runner.return_value = ["native"]
    mock_registry = mock.MagicMock()
    mock_registry.get_all_tasks.return_value = ["custom_t1"]
    mock_global_registry.return_value = mock_registry

    p = pipeline.EvalPipeline(
        model="test_model",
        tasks=("custom_t1",),
        framework=framework_base.FrameworkType.CUSTOM,
        output_dir=self.output_dir,
    )
    self.assertEqual(p._tasks, ["custom_t1"])

    with self.assertRaisesRegex(ValueError, "Tasks not in allowlist"):
      pipeline.EvalPipeline(
          model="test_model",
          tasks=("custom_t2",),
          framework=framework_base.FrameworkType.CUSTOM,
          output_dir=self.output_dir,
      )

  @mock.patch("model_eval.api.pipeline._load_task_allowlist")
  @mock.patch("model_eval.runners.registry.create_runner")
  @mock.patch("model_eval.frameworks.registry.get_framework")
  def test_run_produces_jsonl_files_empty_per_sample(
      self, mock_get_framework, mock_create_runner, mock_load
  ):
    mock_load.return_value = ["mmlu"]
    mock_runner = mock.MagicMock()
    mock_create_runner.return_value = mock_runner
    mock_framework = mock.MagicMock()
    mock_get_framework.return_value = mock_framework

    mock_results = framework_base.EvalResults(
        framework_type="lm-eval",
        aggregated_metrics={"mmlu": {"acc": 0.5}},
        per_sample_outputs={},
    )
    mock_framework.evaluate.return_value = mock_results

    p = pipeline.EvalPipeline(
        model=litert_lm.LiteRtLmRunner.Config(
            runner_type="litert-lm",
            model_path="test_model",
        ),
        tasks=("mmlu",),
        framework=framework_base.FrameworkType.LM_EVAL,
        output_dir=self.output_dir,
    )

    returned_results = p.run()
    self.assertEqual(returned_results, mock_results)

    # Check files created - metrics.jsonl only.
    files = os.listdir(self.output_dir)
    metric_files = [f for f in files if f.endswith("_metrics.jsonl")]
    sample_files = [f for f in files if f.endswith("_samples.jsonl")]

    self.assertEqual(len(metric_files), 1)
    self.assertEqual(len(sample_files), 0)

  @mock.patch("model_eval.api.pipeline._load_task_allowlist")
  @mock.patch("model_eval.frameworks.registry.get_framework")
  def test_run_native_produces_jsonl_files(
      self, mock_get_framework, mock_load
  ):
    """Verifies that running the native path produces metrics and samples JSONL files."""
    mock_load.return_value = ["example_task"]
    mock_framework = mock.MagicMock()
    mock_get_framework.return_value = mock_framework

    mock_results = framework_base.EvalResults(
        framework_type="lm-eval",
        aggregated_metrics={"example_task": {"acc": 0.8}},
        per_sample_outputs={
            "example_task": [{"doc": "test native", "target": "B", "res": "B"}]
        },
    )
    mock_framework.evaluate_native.return_value = mock_results

    model = framework_base.NativeModelConfig(
        model="hf", model_args={"pretrained": "gpt2"}
    )
    p = pipeline.EvalPipeline(
        model=model,
        tasks=("example_task",),
        framework=framework_base.FrameworkType.LM_EVAL,
        output_dir=self.output_dir,
    )

    returned_results = p.run()
    self.assertEqual(returned_results, mock_results)
    mock_framework.evaluate_native.assert_called_once_with(
        model_config=model,
        tasks=["example_task"],
        limit=None,
        sample_range=None,
        batch_size=None,
        eval_args={},
    )

    files = os.listdir(self.output_dir)
    metric_files = [f for f in files if f.endswith("_metrics.jsonl")]
    sample_files = [f for f in files if f.endswith("_samples.jsonl")]

    self.assertEqual(len(metric_files), 1)
    self.assertEqual(len(sample_files), 1)

    with open(os.path.join(self.output_dir, metric_files[0]), "r") as f:
      lines = f.readlines()
      self.assertEqual(len(lines), 1)
      data = json.loads(lines[0])
      self.assertEqual(data["task"], "example_task")
      self.assertEqual(data["metric"], "acc")
      self.assertEqual(data["value"], 0.8)

    with open(os.path.join(self.output_dir, sample_files[0]), "r") as f:
      lines = f.readlines()
      self.assertEqual(len(lines), 1)
      data = json.loads(lines[0])
      self.assertEqual(data["task"], "example_task")
      self.assertEqual(data["doc"], "test native")
      self.assertEqual(data["res"], "B")

  @mock.patch(
      "model_eval.api.pipeline._load_runner_allowlist"
  )
  @mock.patch("model_eval.api.pipeline._load_task_allowlist")
  def test_runner_resolution_falls_back_to_native(
      self, mock_load_task, mock_load_runner
  ):
    """Verifies runner resolution when model has neither runner_type nor model attributes."""
    mock_load_task.return_value = ["mmlu"]
    mock_load_runner.return_value = ["native"]

    class GenericModel:
      pass

    p = pipeline.EvalPipeline(
        model=GenericModel(),
        tasks=("mmlu",),
        framework=framework_base.FrameworkType.LM_EVAL,
        output_dir=self.output_dir,
    )
    self.assertEqual(p._tasks, ["mmlu"])

  @mock.patch(
      "model_eval.api.pipeline._load_runner_allowlist"
  )
  @mock.patch("model_eval.api.pipeline._load_task_allowlist")
  def test_constructor_empty_tasks(self, mock_load_task, mock_load_runner):
    """Verifies that the pipeline can be initialized with empty tasks list."""
    mock_load_task.return_value = ["mmlu"]
    mock_load_runner.return_value = ["native"]

    p = pipeline.EvalPipeline(
        model="native",
        tasks=(),
        framework=framework_base.FrameworkType.LM_EVAL,
        output_dir=self.output_dir,
    )
    self.assertEqual(p._tasks, [])


if __name__ == "__main__":
  absltest.main()
