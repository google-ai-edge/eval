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

"""Unit tests for the http-server runner."""

from absl.testing import absltest
from absl.testing import parameterized
from model_eval.cli import main
from model_eval.runners import base
from model_eval.runners import http_server
from model_eval.runners import registry

_TIMEOUT_ERROR = "request_timeout_sec must be a positive number"


class HttpServerConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("base_url", "http://10.0.0.1:8080", "http://10.0.0.1:8080"),
      ("trailing_slash", "http://10.0.0.1:8080/", "http://10.0.0.1:8080"),
      ("https", "https://host", "https://host"),
      ("path_prefix", "http://host/prefix/", "http://host/prefix"),
  )
  def test_server_url(self, server_url, expected):
    config = http_server.HttpServerRunner.Config.from_unified_args(
        None,
        None,
        {"server_url": server_url},
    )
    self.assertEqual(config.server_url, expected)
    self.assertEqual(config.runner_type, base.RunnerType.HTTP_SERVER)

  @parameterized.named_parameters(
      ("no_scheme", "10.0.0.1:8080", "must be an http"),
      ("bad_scheme", "ftp://host", "must be an http"),
      ("no_host", "http://", "must be an http"),
  )
  def test_rejects_invalid_server_url(self, server_url, error):
    with self.assertRaisesRegex(ValueError, error):
      http_server.HttpServerRunner.Config.from_unified_args(
          None,
          None,
          {"server_url": server_url},
      )

  def test_ignores_model_path_and_device(self):
    config = http_server.HttpServerRunner.Config.from_unified_args(
        "/path/to/model",
        "gpu",
        {"server_url": "http://host:1"},
    )
    self.assertEqual(config.server_url, "http://host:1")
    self.assertEqual(config.server_args, {})

  def test_collects_unknown_runner_args_as_server_args(self):
    config = http_server.HttpServerRunner.Config.from_unified_args(
        None,
        None,
        {
            "server_url": "http://host:1",
            "request_timeout_sec": 10,
            "model_name": "gemma-4-E2B-it",
            "api_version": 2,
        },
    )
    self.assertEqual(config.request_timeout_sec, 10.0)
    self.assertEqual(
        config.server_args, {"model_name": "gemma-4-E2B-it", "api_version": 2}
    )

  @parameterized.named_parameters(
      ("missing", {}),
      ("empty", {"server_url": ""}),
  )
  def test_requires_server_url(self, runner_args):
    with self.assertRaisesRegex(ValueError, "server_url is required"):
      http_server.HttpServerRunner.Config.from_unified_args(
          None, None, runner_args
      )

  def test_cli_builds_http_server_config(self):
    config = main._build_model_config(
        "http-server",
        "server_url=http://host:1/,model_name=gemma-4-E2B-it",
    )
    self.assertIsInstance(config, http_server.HttpServerRunner.Config)
    self.assertEqual(config.server_url, "http://host:1")
    self.assertIsNone(config.request_timeout_sec)
    self.assertEqual(config.server_args, {"model_name": "gemma-4-E2B-it"})

  def test_cli_builds_timeout(self):
    config = main._build_model_config(
        "http-server",
        "server_url=http://host:1,request_timeout_sec=1800",
    )
    self.assertEqual(config.request_timeout_sec, 1800.0)

  @parameterized.named_parameters(
      ("zero_timeout", {"request_timeout_sec": 0}, _TIMEOUT_ERROR),
      ("negative_timeout", {"request_timeout_sec": -1}, _TIMEOUT_ERROR),
      ("text_timeout", {"request_timeout_sec": "soon"}, _TIMEOUT_ERROR),
      ("bool_timeout", {"request_timeout_sec": True}, _TIMEOUT_ERROR),
  )
  def test_rejects_invalid_timeout(self, extra_args, error):
    with self.assertRaisesRegex(ValueError, error):
      http_server.HttpServerRunner.Config.from_unified_args(
          None, None, {"server_url": "http://host:1", **extra_args}
      )


class HttpServerRunnerTest(absltest.TestCase):

  def _runner(self, **extra_config):
    return registry.create_runner(
        base.RunnerType.HTTP_SERVER,
        {
            "runner_type": base.RunnerType.HTTP_SERVER,
            "server_url": "http://host:1",
            **extra_config,
        },
    )

  def test_registered_and_exposes_config(self):
    runner = self._runner()
    self.assertIsInstance(runner, http_server.HttpServerRunner)
    self.assertIs(
        registry.get_runner_cls(base.RunnerType.HTTP_SERVER),
        http_server.HttpServerRunner,
    )
    self.assertEqual(runner.server_url, "http://host:1")
    self.assertIsNone(runner.request_timeout_sec)

  def test_exposes_timeout(self):
    runner = self._runner(request_timeout_sec=1800.0)
    self.assertEqual(runner.request_timeout_sec, 1800.0)
    self.assertEqual(runner.server_args, {})

  def test_exposes_server_args_copy(self):
    runner = self._runner(server_args={"model_name": "gemma-4-E2B-it"})
    args = runner.server_args
    self.assertEqual(args, {"model_name": "gemma-4-E2B-it"})
    args["model_name"] = "mutated"
    self.assertEqual(runner.server_args, {"model_name": "gemma-4-E2B-it"})

  def test_start_and_stop_are_noops(self):
    runner = self._runner()
    with runner as entered:
      self.assertIs(entered, runner)
    runner.start()
    runner.stop()

  def test_describes_runner_args(self):
    names = [
        arg["name"]
        for arg in http_server.HttpServerRunner.describe_runner_args()
    ]
    self.assertIn("server_url", names)
    self.assertIn("request_timeout_sec", names)
    self.assertIn("server_args", names)


if __name__ == "__main__":
  absltest.main()
