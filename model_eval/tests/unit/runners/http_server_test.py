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
_ENDPOINTS_ERROR = "endpoints must be a non-empty path or list of paths"


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
        {"server_url": server_url, "endpoints": "v1/chat/completions"},
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
          {"server_url": server_url, "endpoints": "v1/chat/completions"},
      )

  @parameterized.named_parameters(
      ("single_str", "v1/embeddings", ("v1/embeddings",)),
      ("single_with_slashes", "/v1/embeddings/", ("v1/embeddings",)),
      (
          "list_of_paths",
          ["/v1/chat/completions/", "v1/chat/score"],
          ("v1/chat/completions", "v1/chat/score"),
      ),
      ("tuple_of_paths", ("a", "/b/"), ("a", "b")),
  )
  def test_endpoints(self, endpoints, expected):
    config = http_server.HttpServerRunner.Config.from_unified_args(
        None, None, {"server_url": "http://host:1", "endpoints": endpoints}
    )
    self.assertEqual(config.endpoints, expected)

  def test_ignores_model_path_and_device(self):
    config = http_server.HttpServerRunner.Config.from_unified_args(
        "/path/to/model",
        "gpu",
        {"server_url": "http://host:1", "endpoints": "v1/embeddings"},
    )
    self.assertEqual(config.server_url, "http://host:1")
    self.assertEqual(config.endpoints, ("v1/embeddings",))

  @parameterized.named_parameters(
      ("missing", {"endpoints": "v1/embeddings"}),
      ("empty", {"server_url": "", "endpoints": "v1/embeddings"}),
  )
  def test_requires_server_url(self, runner_args):
    with self.assertRaisesRegex(ValueError, "server_url is required"):
      http_server.HttpServerRunner.Config.from_unified_args(
          None, None, runner_args
      )

  def test_requires_endpoints(self):
    with self.assertRaisesRegex(ValueError, "endpoints is required"):
      http_server.HttpServerRunner.Config.from_unified_args(
          None, None, {"server_url": "http://host:1"}
      )

  def test_cli_builds_http_server_config(self):
    config = main._build_model_config(
        "http-server",
        "server_url=http://host:1/,"
        "endpoints=['/v1/chat/completions','v1/chat/score']",
    )
    self.assertIsInstance(config, http_server.HttpServerRunner.Config)
    self.assertEqual(config.server_url, "http://host:1")
    self.assertEqual(
        config.endpoints, ("v1/chat/completions", "v1/chat/score")
    )
    self.assertIsNone(config.request_timeout_sec)

  def test_cli_builds_single_endpoint_and_timeout(self):
    config = main._build_model_config(
        "http-server",
        "server_url=http://host:1,endpoints=/v1/embeddings/,"
        "request_timeout_sec=1800",
    )
    self.assertEqual(config.endpoints, ("v1/embeddings",))
    self.assertEqual(config.request_timeout_sec, 1800.0)

  @parameterized.named_parameters(
      (
          "empty_str",
          {"endpoints": ""},
          "Each endpoint must be a non-empty path",
      ),
      (
          "slash_str",
          {"endpoints": "/"},
          "Each endpoint must be a non-empty path",
      ),
      (
          "empty_list",
          {"endpoints": []},
          "endpoints must contain at least one path",
      ),
      (
          "list_with_empty",
          {"endpoints": ["a", "/"]},
          "Each endpoint must be a non-empty path",
      ),
      (
          "list_with_int",
          {"endpoints": [1]},
          "Each endpoint must be a non-empty string",
      ),
      (
          "non_sequence",
          {"endpoints": 123},
          "endpoints must be a path string or a list of path strings",
      ),
      (
          "zero_timeout",
          {"endpoints": "op", "request_timeout_sec": 0},
          _TIMEOUT_ERROR,
      ),
      (
          "negative_timeout",
          {"endpoints": "op", "request_timeout_sec": -1},
          _TIMEOUT_ERROR,
      ),
      (
          "text_timeout",
          {"endpoints": "op", "request_timeout_sec": "soon"},
          _TIMEOUT_ERROR,
      ),
      (
          "bool_timeout",
          {"endpoints": "op", "request_timeout_sec": True},
          _TIMEOUT_ERROR,
      ),
  )
  def test_rejects_invalid_endpoints_and_timeout(self, extra_args, error):
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
            "endpoints": ("v1/chat/completions",),
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
    self.assertEqual(runner.endpoints, ("v1/chat/completions",))
    self.assertIsNone(runner.request_timeout_sec)

  def test_exposes_endpoints_and_timeout(self):
    runner = self._runner(
        endpoints=("v1/embeddings",), request_timeout_sec=1800.0
    )
    self.assertEqual(runner.endpoints, ("v1/embeddings",))
    self.assertEqual(runner.request_timeout_sec, 1800.0)

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
    self.assertIn("endpoints", names)
    self.assertIn("request_timeout_sec", names)


if __name__ == "__main__":
  absltest.main()
