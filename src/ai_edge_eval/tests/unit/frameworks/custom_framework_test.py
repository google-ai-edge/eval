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

"""Unit tests for CustomFramework."""

from unittest import mock
from absl.testing import absltest
from ai_edge_eval import config
from ai_edge_eval import custom_tasks as tasks
from ai_edge_eval.frameworks import custom


class CustomFrameworkTest(absltest.TestCase):

  def test_generate_sends_payload_correctly(self):
    runner = mock.MagicMock()
    runner.server_url = "http://dummy"
    runner.model_name = "fake-model"

    task = tasks.CustomTask(
        name="foo",
        dataset="dummy.jsonl",
        metric_fn=lambda p, g, r: {},
        generation_config=config.GenerationConfig(),
    )

    framework = custom.CustomFramework()

    # Mock POST response.
    mock_response = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "generated content"}}]
    }
    mock_client = mock.MagicMock()
    mock_client.post = mock.MagicMock(return_value=mock_response)

    input_msgs = [{"role": "user", "content": "hello"}]

    pred_text = framework._generate(
        runner, input_msgs, task.generation_config, mock_client
    )

    # Check post calls.
    mock_client.post.assert_called_once_with(
        "http://dummy/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": input_msgs,
            "temperature": 1.0,
            "max_tokens": 256,
            "stop": None,
        },
    )
    self.assertEqual(pred_text, "generated content")

  def test_apply_limit(self):
    rows = [{"id": i} for i in range(10)]
    self.assertEqual(custom._apply_limit(rows, 3), rows[:3])
    self.assertEqual(custom._apply_limit(rows, 0.5), rows[:5])

  def test_apply_samples(self):
    rows = [{"id": i} for i in range(5)]
    # Test dict mapping for specific task.
    res_dict = custom._apply_samples(rows, {"foo": "1-3"}, "foo")
    self.assertEqual(res_dict, [{"id": 1}, {"id": 2}, {"id": 3}])

    # Test list of specific indices.
    res_list = custom._apply_samples(rows, [0, 4], "foo")
    self.assertEqual(res_list, [{"id": 0}, {"id": 4}])

  def test_generate_with_gen_kwargs(self):
    runner = mock.MagicMock()
    runner.server_url = "http://dummy"
    runner.model_name = "fake"

    cfg = config.GenerationConfig(
        temperature=0.2, max_new_tokens=64, stop_sequences=["\n", "END"]
    )
    task = tasks.CustomTask(
        name="foo",
        dataset="dummy.jsonl",
        metric_fn=lambda p, g, r: {},
        generation_config=cfg,
    )
    framework = custom.CustomFramework()
    mock_response = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "yes"}}]
    }
    mock_client = mock.MagicMock()
    mock_client.post = mock.MagicMock(return_value=mock_response)

    input_msgs = [{"role": "user", "content": "?"}]
    pred_text = framework._generate(
        runner, input_msgs, task.generation_config, mock_client
    )

    mock_client.post.assert_called_once_with(
        "http://dummy/v1/chat/completions",
        json={
            "model": "fake",
            "messages": input_msgs,
            "temperature": 0.2,
            "max_tokens": 64,
            "stop": ["\n", "END"],
        },
    )
    self.assertEqual(pred_text, "yes")

  def test_evaluate_raises_value_error_on_both_limit_and_samples(self):
    framework = custom.CustomFramework()
    runner = mock.MagicMock()
    with self.assertRaisesRegex(
        ValueError, "Only one of 'limit' or 'samples' can be set, not both."
    ):
      framework.evaluate(
          runner, tasks=["foo"], limit=10, eval_args={"samples": "0-5"}
      )

  def test_run_task_raises_value_error_on_both_limit_and_samples(self):
    framework = custom.CustomFramework()
    runner = mock.MagicMock()
    task = mock.MagicMock()
    mock_client = mock.MagicMock()
    with self.assertRaisesRegex(
        ValueError, "Only one of 'limit' or 'samples' can be set, not both."
    ):
      framework._run_task(runner, task, mock_client, limit=10, samples="0-5")

  def test_evaluate_conflicts(self):
    framework = custom.CustomFramework()
    runner = mock.MagicMock()
    with self.assertRaisesRegex(
        ValueError, "--limit conflicts with 'limit' in --eval-args"
    ):
      framework.evaluate(runner, ["foo"], limit=5, eval_args={"limit": 5})

    with self.assertRaisesRegex(
        ValueError, "--batch-size conflicts with 'batch_size' in --eval-args"
    ):
      framework.evaluate(
          runner, ["foo"], batch_size=2, eval_args={"batch_size": 2}
      )

  def test_describe_eval_args(self):
    fields = custom.CustomFramework.describe_eval_args()
    self.assertIsInstance(fields, list)
    for field in fields:
      self.assertIn("name", field)

  @mock.patch("ai_edge_eval.runners.registry.get_all_runners")
  def test_supported_runners(self, mock_get_all_runners):
    mock_get_all_runners.return_value = ["runner1", "runner2"]
    runners = custom.CustomFramework.supported_runners()
    self.assertEqual(runners, ["runner1", "runner2"])


if __name__ == "__main__":
  absltest.main()
