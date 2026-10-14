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

import os
import tempfile
import unittest
from unittest import mock

from absl.testing import parameterized
from model_eval.cli import main
from model_eval.frameworks import base as framework_base
from model_eval.runners import litert_lm
from click import testing


class MainTest(parameterized.TestCase):

  @parameterized.parameters((True,), (False,))
  @mock.patch("model_eval.cli.main.eval_pipeline.EvalPipeline")
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

  @mock.patch("model_eval.cli.main.eval_pipeline.EvalPipeline")
  def test_from_samples_runs_without_runner(self, mock_pipeline_cls):
    mock_pipeline = mock.MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline
    mock_pipeline.run.return_value.aggregated_metrics = {}
    with tempfile.TemporaryDirectory() as temp_dir:
      samples = os.path.join(temp_dir, "samples.jsonl")
      open(samples, "w").close()
      result = testing.CliRunner().invoke(
          main.cli,
          [
              "--framework",
              "custom",
              "--tasks",
              "t",
              "--from-samples",
              samples,
              "--eval-args",
              "metrics=wer",
          ],
      )
    self.assertEqual(result.exit_code, 0, result.output)
    _, kwargs = mock_pipeline_cls.call_args
    self.assertIsNone(kwargs["model"])
    self.assertEqual(
        kwargs["eval_args"], {"metrics": "wer", "from_samples": samples}
    )

  @mock.patch("model_eval.cli.main.eval_pipeline.EvalPipeline")
  def test_from_samples_accepts_remote_cns_path_and_eval_args(
      self, mock_pipeline_cls
  ):
    mock_pipeline = mock.MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline
    mock_pipeline.run.return_value.aggregated_metrics = {}
    cns_path = "/cns/cell/home/user/run_123_samples.jsonl"
    result = testing.CliRunner().invoke(
        main.cli,
        [
            "--framework",
            "custom",
            "--tasks",
            "t",
            "--from-samples",
            cns_path,
        ],
    )
    self.assertEqual(result.exit_code, 0, result.output)
    _, kwargs = mock_pipeline_cls.call_args
    self.assertIsNone(kwargs["model"])
    self.assertEqual(kwargs["eval_args"], {"from_samples": cns_path})

    mock_pipeline_cls.reset_mock()
    result_via_eval_args = testing.CliRunner().invoke(
        main.cli,
        [
            "--framework",
            "custom",
            "--tasks",
            "t",
            "--eval-args",
            f"from_samples={cns_path},metrics=wer",
        ],
    )
    self.assertEqual(
        result_via_eval_args.exit_code, 0, result_via_eval_args.output
    )
    _, kwargs = mock_pipeline_cls.call_args
    self.assertIsNone(kwargs["model"])
    self.assertEqual(
        kwargs["eval_args"], {"from_samples": cns_path, "metrics": "wer"}
    )

  @mock.patch("model_eval.cli.main.eval_pipeline.EvalPipeline")
  def test_inference_only_sets_eval_arg(self, mock_pipeline_cls):
    mock_pipeline = mock.MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline
    mock_pipeline.run.return_value.aggregated_metrics = {}
    result = testing.CliRunner().invoke(
        main.cli,
        [
            "--runner",
            "http-server",
            "--runner-args",
            "server_url=http://h:1,endpoints=op",
            "--framework",
            "custom",
            "--tasks",
            "t",
            "--inference-only",
            "--eval-args",
            "endpoint=op",
        ],
    )
    self.assertEqual(result.exit_code, 0, repr(result.exception))
    _, kwargs = mock_pipeline_cls.call_args
    self.assertEqual(
        kwargs["eval_args"], {"endpoint": "op", "inference_only": True}
    )

  def test_staging_flags_usage_errors(self):
    runner = testing.CliRunner()
    with tempfile.TemporaryDirectory() as temp_dir:
      samples = os.path.join(temp_dir, "samples.jsonl")
      open(samples, "w").close()
      result = runner.invoke(
          main.cli,
          [
              "--framework",
              "custom",
              "--tasks",
              "t",
              "--from-samples",
              samples,
              "--inference-only",
          ],
      )
      self.assertNotEqual(result.exit_code, 0)
      self.assertIn("mutually exclusive", result.output)
      result = runner.invoke(
          main.cli,
          ["--framework", "lm-eval", "--tasks", "t", "--from-samples", samples],
      )
      self.assertNotEqual(result.exit_code, 0)
      self.assertIn("require --framework custom", result.output)
    result = runner.invoke(
        main.cli,
        [
            "--framework",
            "lm-eval",
            "--runner",
            "litert-lm",
            "--tasks",
            "t",
            "--eval-args",
            "inference_only=true",
        ],
    )
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("require --framework custom", result.output)
    result = runner.invoke(main.cli, ["--framework", "custom", "--tasks", "t"])
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("Missing option '--runner'", result.output)

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
  @mock.patch("model_eval.cli.main.eval_pipeline.EvalPipeline")
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

  @mock.patch("sys.platform", "darwin")
  @mock.patch("multiprocessing.set_start_method")
  def test_multiprocessing_set_start_method_fork_on_mac(
      self, mock_set_start_method
  ):
    import importlib
    # Reload main module to run module-level initialization code
    importlib.reload(main)
    mock_set_start_method.assert_called_once_with("fork", force=True)

  @mock.patch("sys.platform", "linux")
  @mock.patch("multiprocessing.set_start_method")
  def test_multiprocessing_set_start_method_default_on_non_mac(
      self, mock_set_start_method
  ):
    import importlib

    importlib.reload(main)
    mock_set_start_method.assert_not_called()

  def test_run_pipeline_calls_flume_runner_program_started(self):
    fake_flume = mock.MagicMock()
    with (
        mock.patch.object(main, "flume_runner", fake_flume),
        mock.patch.object(main.cli, "main") as mock_cli_main,
    ):
      main._run_pipeline(["cli", "--version"])
    fake_flume.program_started.assert_called_once_with()
    mock_cli_main.assert_called_once_with(
        args=["--version"], standalone_mode=False
    )


if __name__ == "__main__":
  unittest.main()
