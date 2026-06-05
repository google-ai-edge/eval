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

"""Integration tests for LiteRT LM runner against physical models."""

import os
import unittest

from absl.testing import absltest
from model_eval.runners.litert_lm import litert_lm
import requests


class LiteRtLmIntegrationTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    model_path = os.environ.get("AI_EDGE_EVAL_INTEGRATION_TEST_MODEL")
    if not model_path:
      model_path = "google/tiny-gemma-litert"

    self.config = litert_lm.LiteRtLmRunner.Config(
        runner_type="litert-lm",
        model_path=model_path,
        model_name="tiny-gemma-integration",
        backend="cpu",
        host="127.0.0.1",
        port=19191,
        max_num_tokens=1024,
    )
    self.runner = litert_lm.LiteRtLmRunner(self.config)
    self.runner.start()
    self.base_url = f"http://{self.config.host}:{self.config.port}/v1"

  def tearDown(self):
    self.runner.stop()
    super().tearDown()

  def test_chat_completions(self):
    response = requests.post(
        f"{self.base_url}/chat/completions",
        json={
            "model": "tiny-gemma-integration",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "top_k": 10,
            "top_p": 0.9,
            "max_tokens": 10,
        },
        timeout=15,
    )
    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertEqual(data["model"], "tiny-gemma-integration")
    self.assertEqual(data["object"], "chat.completion")
    choices = data["choices"]
    self.assertEqual(len(choices), 1)
    self.assertIsInstance(choices[0]["message"]["content"], str)

  def test_chat_score(self):
    response = requests.post(
        f"{self.base_url}/chat/score",
        json={
            "model": "tiny-gemma-integration",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        },
        timeout=15,
    )
    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertEqual(data["model"], "tiny-gemma-integration")
    self.assertEqual(data["object"], "chat.score")
    choices = data["choices"]
    self.assertEqual(len(choices), 1)
    self.assertIsInstance(choices[0]["score"], float)
    self.assertIn("token_logprobs", choices[0]["logprobs"])


if __name__ == "__main__":
  absltest.main()
