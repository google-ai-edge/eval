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

"""Unit tests for the CLI module."""

import unittest
from unittest import mock

from absl.testing import parameterized
from model_eval.cli import main
from model_eval.frameworks import base as framework_base
from model_eval.runners import litert_lm
from click import testing


class MainTest(parameterized.TestCase):

  @parameterized.parameters((True,), (False,))
  @mock.patch(
      "model_eval.cli.main.eval_pipeline.EvalPipeline"
  )
  def test_main_parses_comma_separated_tasks(
      self, include_eval, mock_pipeline_cls
  ):
    mock_pipeline = mock.MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline
    mock_pipeline.run.return_value.aggregated_metrics = {}

    runner = testing.CliRunner()
    args = [
        "--runner",
        "litert-lm",
        "--model-path",
        "fake.tflite",
        "--device",
        "cpu",
        "--runner-args",
        "vision_backend=gpu,audio_backend=npu",
        "--framework",
        "lm-eval",
        "--tasks",
        "mmlu",
        "--tasks",
        "gsm8k",
        "--tasks",
        "bbh",
        "--eval-args",
        "limit=10,num_fewshot=0,apply_chat_template=true",
        "--output-dir",
        "/tmp/results/",
    ]
    if include_eval:
      args.append("eval")

    result = runner.invoke(main.cli, args)

    self.assertEqual(result.exit_code, 0)
    mock_pipeline_cls.assert_called_once()
    _, kwargs = mock_pipeline_cls.call_args
    self.assertEqual(kwargs["tasks"], ("mmlu", "gsm8k", "bbh"))
    mock_pipeline.run.assert_called_once()

  def test_parse_csv_args_parses_json_values(self):
    self.assertEqual(
        main._parse_csv_args('limit=10,extra={"a": 1}'),
        {"limit": 10, "extra": {"a": 1}},
    )

  def test_parse_csv_args_fallback_to_string(self):
    self.assertEqual(
        main._parse_csv_args("path=/my/file,limit=10"),
        {"path": "/my/file", "limit": 10},
    )

  def test_build_model_config_litert_runner(self):
    config = main._build_model_config("litert-lm", "model_path=fake.bin")
    self.assertIsInstance(config, litert_lm.LiteRtLmRunner.Config)
    self.assertEqual(config.model_path, "fake.bin")

  def test_build_model_config_native(self):
    config = main._build_model_config("hf", "pretrained=gpt2")
    self.assertIsInstance(config, framework_base.NativeModelConfig)
    self.assertEqual(config.model, "hf")
    self.assertEqual(config.model_args, {"pretrained": "gpt2"})

  def test_parse_csv_args_nested_quotes(self):
    res = main._parse_csv_args("limit=10,samples={'mmlu': [0, 1, 2]}")
    self.assertEqual(res, {"limit": 10, "samples": {"mmlu": [0, 1, 2]}})

  def test_parse_csv_args_list_of_dicts(self):
    res = main._parse_csv_args("data=[{'a': 1}, {'b': 2}]")
    self.assertEqual(res, {"data": [{"a": 1}, {"b": 2}]})

  @mock.patch("model_eval.cli.main.importlib.import_module")
  @mock.patch(
      "model_eval.cli.main.eval_pipeline.EvalPipeline"
  )
  def test_custom_tasks_file_triggers_import(
      self, mock_pipeline_cls, mock_import
  ):
    mock_pipeline = mock.MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline
    mock_pipeline.run.return_value.aggregated_metrics = {}

    runner = testing.CliRunner()
    result = runner.invoke(
        main.cli,
        [
            "--runner",
            "litert-lm",
            "--runner-args",
            "model_path=fake.tflite",
            "--framework",
            "lm-eval",
            "--tasks",
            "mmlu",
            "--custom-tasks-file",
            "fake_module.py",
        ],
    )
    self.assertEqual(result.exit_code, 0)
    mock_import.assert_called_once_with("fake_module")

  def test_list_runners_with_custom_config(self):
    """Verifies that list-runners command respects custom runner config."""
    import os
    import tempfile
    import yaml

    with tempfile.TemporaryDirectory() as temp_dir:
      custom_config = {"lm_eval": ["my_custom_runner"]}
      config_path = os.path.join(temp_dir, "custom_runners.yaml")
      with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(custom_config, f)

      runner = testing.CliRunner()
      result = runner.invoke(
          main.cli,
          [
              "list-runners",
              "--framework",
              "lm-eval",
              "--runner-config",
              config_path,
          ],
      )

      self.assertEqual(result.exit_code, 0)
      self.assertIn("Supported runners for framework 'lm-eval':", result.output)
      self.assertIn("- my_custom_runner", result.output)


if __name__ == "__main__":
  unittest.main()
