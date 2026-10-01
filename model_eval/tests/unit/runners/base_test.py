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

"""Unit tests for base runner."""

import unittest
from unittest import mock

from model_eval.api import constants
from model_eval.runners import base


class DummyRunner(base.AbstractRunner):

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  @property
  def server_url(self) -> str:
    return "http://127.0.0.1:8080"

  @property
  def endpoints(self) -> tuple[str, ...]:
    return (constants.CHAT_COMPLETIONS_ENDPOINT, constants.CHAT_SCORE_ENDPOINT)

  @property
  def request_timeout_sec(self) -> float | None:
    return None

  class Config(base.RunnerConfig):

    @classmethod
    def from_unified_args(
        cls, model_path, device, runner_args
    ) -> "DummyRunner.Config":
      del model_path, device
      return cls(runner_type="dummy", **runner_args)


class TestBaseRunner(unittest.TestCase):

  def test_runner_config_request_timeout_sec(self):
    self.assertIsNone(
        DummyRunner.Config(runner_type="dummy").request_timeout_sec
    )
    self.assertEqual(
        DummyRunner.Config.from_unified_args(
            None, None, {"request_timeout_sec": 30}
        ).request_timeout_sec,
        30.0,
    )
    for bad in (0, -1, "soon", True, False):
      with self.subTest(bad=bad):
        with self.assertRaisesRegex(
            ValueError, "request_timeout_sec must be a positive number"
        ):
          DummyRunner.Config(runner_type="dummy", request_timeout_sec=bad)

  def test_describe_runner_args(self):
    names = [arg["name"] for arg in DummyRunner.describe_runner_args()]
    self.assertIn("request_timeout_sec", names)

  def test_reentrancy_guard(self):
    runner = DummyRunner()
    self.assertEqual(runner.server_url, "http://127.0.0.1:8080")
    self.assertEqual(
        runner.endpoints,
        (constants.CHAT_COMPLETIONS_ENDPOINT, constants.CHAT_SCORE_ENDPOINT),
    )
    self.assertIsNone(runner.request_timeout_sec)
    # Exiting before entering is a safe no-op.
    runner.__exit__()
    with mock.patch.object(
        runner, "start", wraps=runner.start
    ) as mock_start, mock.patch.object(
        runner, "stop", wraps=runner.stop
    ) as mock_stop:

      # Enter recursively
      with runner:
        self.assertEqual(runner._ref_count, 1)
        with runner:
          self.assertEqual(runner._ref_count, 2)
          # Server should only start once
          mock_start.assert_called_once()
          mock_stop.assert_not_called()

        # After inner exit, server should still be running (ref_count = 1)
        self.assertEqual(runner._ref_count, 1)
        mock_stop.assert_not_called()

      # After outer exit, server should stop (ref_count = 0)
      self.assertEqual(runner._ref_count, 0)
      mock_start.assert_called_once()
      mock_stop.assert_called_once()


if __name__ == "__main__":
  unittest.main()
