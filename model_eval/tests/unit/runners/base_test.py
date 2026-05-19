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
  def model_name(self) -> str:
    return "dummy_model"

  @property
  def capabilities(self) -> base.RunnerCapabilities:
    return base.RunnerCapabilities()

  @classmethod
  def from_unified_args(
      cls, model_path, device, runner_args
  ) -> base.RunnerConfig:
    return base.RunnerConfig(runner_type="dummy")


class TestBaseRunner(unittest.TestCase):

  @mock.patch("model_eval.runners.base.requests.post")
  def test_validate_completions_success(self, mock_post):
    mock_post.return_value.json.return_value = {"choices": []}
    runner = DummyRunner()
    runner._validate_completions()
    mock_post.assert_called_once_with(
        f"http://127.0.0.1:8080/{constants.CHAT_COMPLETIONS_ENDPOINT}",
        json={
            "model": "dummy_model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
        timeout=20,
    )

  @mock.patch("model_eval.runners.base.requests.post")
  def test_validate_completions_failure(self, mock_post):
    mock_post.return_value.json.return_value = {}
    runner = DummyRunner()
    with self.assertRaisesRegex(
        RuntimeError, "Runner failed generation validation"):
      runner._validate_completions()

  @mock.patch("model_eval.runners.base.requests.post")
  def test_validate_scoring_success(self, mock_post):
    mock_post.return_value.json.return_value = {
        "choices": [{"score": 0.9, "logprobs": []}]
    }
    runner = DummyRunner()
    runner._validate_scoring()
    mock_post.assert_called_once_with(
        f"http://127.0.0.1:8080/{constants.CHAT_SCORE_ENDPOINT}",
        json={
            "model": "dummy_model",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        },
        timeout=20,
    )

  @mock.patch("model_eval.runners.base.requests.post")
  def test_validate_scoring_failure(self, mock_post):
    mock_post.return_value.json.return_value = {"choices": [{}]}
    runner = DummyRunner()
    with self.assertRaisesRegex(
        RuntimeError, "Runner failed scoring validation"):
      runner._validate_scoring()


if __name__ == "__main__":
  unittest.main()
