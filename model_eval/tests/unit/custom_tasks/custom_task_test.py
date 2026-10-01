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
from model_eval import config
from model_eval import custom_tasks as tasks


class CustomTaskTest(absltest.TestCase):

  def test_construction(self):
    def dummy_metric(preds, gts, rows):
      return {"val": 1.0}

    task = tasks.CustomTask(
        name="t1", dataset="f.jsonl", metric_fn=dummy_metric
    )
    self.assertEqual(task.name, "t1")
    self.assertEqual(task.dataset, "f.jsonl")
    self.assertIs(task.metric_fn, dummy_metric)

  def test_requests_alias_and_dataset_row_metadata(self):
    reqs: tasks.Requests = [
        {"op": "index", "video_path": "v.mp4"},
        {"op": "query", "query": "hello"},
    ]
    row_without_metadata: tasks.DatasetRow[int] = {
        "requests": reqs,
        "ground_truth": 1,
    }
    self.assertNotIn("metadata", row_without_metadata)

    row_with_metadata: tasks.DatasetRow[int] = {
        "requests": reqs,
        "ground_truth": 1,
        "metadata": {"source": "test_data"},
    }
    self.assertIn("metadata", row_with_metadata)
    self.assertEqual(row_with_metadata["metadata"]["source"], "test_data")

  def test_chat_request_defaults_and_custom_config(self):
    messages = [{"role": "user", "content": "hello"}]
    self.assertEqual(
        tasks.chat_request(messages),
        {
            "model": "default_model",
            "messages": messages,
            "temperature": 1.0,
            "max_tokens": 256,
            "stop": None,
        },
    )
    cfg = config.GenerationConfig(
        temperature=0.2, max_new_tokens=64, stop_sequences=["\n", "END"]
    )
    self.assertEqual(
        tasks.chat_request(messages, cfg),
        {
            "model": "default_model",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 64,
            "stop": ["\n", "END"],
        },
    )

  def test_chat_response_text_extracts_content_or_returns_none(self):
    self.assertEqual(
        tasks.chat_response_text(
            {"choices": [{"message": {"content": "generated text"}}]}
        ),
        "generated text",
    )
    self.assertIsNone(tasks.chat_response_text(None))


if __name__ == "__main__":
  absltest.main()
