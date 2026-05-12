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

"""Unit tests for CustomTask definitions."""

from absl.testing import absltest
from ai_edge_eval import config
from ai_edge_eval import custom_tasks as tasks


class CustomTaskTest(absltest.TestCase):

  def test_default_construction(self):
    def dummy_metric(preds, gts, rows):
      return {"val": 1.0}

    task = tasks.CustomTask(
        name="t1", dataset="f.jsonl", metric_fn=dummy_metric
    )
    self.assertEqual(task.name, "t1")
    self.assertEqual(task.dataset, "f.jsonl")
    self.assertEqual(task.generation_config.temperature, 1.0)

  def test_custom_construction(self):
    cfg = config.GenerationConfig(temperature=0.5)
    task = tasks.CustomTask(
        name="t2",
        dataset="f.csv",
        metric_fn=lambda p, g, r: {},
        generation_config=cfg,
    )
    self.assertEqual(task.generation_config.temperature, 0.5)

  def test_openai_messages_alias(self):
    sample_message: tasks.OpenAIMessages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    self.assertIsInstance(sample_message, list)
    self.assertLen(sample_message, 2)
    self.assertEqual(sample_message[0]["role"], "user")

  def test_dataset_row_metadata(self):
    row_without_metadata: tasks.DatasetRow[int] = {
        "messages": [{"role": "user", "content": "hello"}],
        "ground_truth": 1,
    }
    self.assertNotIn("metadata", row_without_metadata)

    row_with_metadata: tasks.DatasetRow[int] = {
        "messages": [{"role": "user", "content": "hello"}],
        "ground_truth": 1,
        "metadata": {"source": "test_data"},
    }
    self.assertIn("metadata", row_with_metadata)
    self.assertEqual(row_with_metadata["metadata"]["source"], "test_data")



if __name__ == "__main__":
  absltest.main()
