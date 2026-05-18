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

r"""Example script demonstrating how to register a custom evaluation task.

Example command:
```bash
ai-edge-eval \
  --runner litert-lm \
  --model-path /tmp/gemma3-270m-it-q8.litertlm \
  --device cpu \
  --framework custom \
  --eval-args "samples={'my_iterator_qa':'1-3'}" \
  --custom-tasks-file examples/custom_task_example.py \
  --tasks my_iterator_qa \
  --output-dir /tmp/results
```
"""

from typing import Iterator

from ai_edge_eval import config
from ai_edge_eval import custom_tasks

OpenAIMessages = custom_tasks.OpenAIMessages


def my_custom_iterator() -> Iterator[custom_tasks.DatasetRow]:
  """Generator that yields DatasetRow (input + ground truth)."""
  dataset = [
      {"q": "What is the capital of France?", "a": "Paris"},
      {"q": "What is 2+2?", "a": "4"},
      {"q": "Who wrote Hamlet?", "a": "Shakespeare"},
      {"q": "What is the color of Mars?", "a": "Red"},
  ]
  for row in dataset:
    yield {
        "messages": [{"role": "user", "content": row["q"]}],
        "ground_truth": row["a"],
    }


def text_metrics(
    preds: list[str],
    groundtruths: list[str],
    rows: list[custom_tasks.DatasetRow],
):
  """Evaluates if the ground truth is contained within the model's output."""
  del rows  # Unused.
  pred_texts = [p.lower() for p in preds]
  groundtruth_texts = [g.lower() for g in groundtruths]

  # A match is recorded if the reference string 'gt_text' is found inside
  # 'pred_text'.
  hits = sum(
      1
      for pred_text, gt_text in zip(pred_texts, groundtruth_texts)
      if gt_text in pred_text
  )
  reference_in_sample_accuracy = hits / len(pred_texts)
  # A match is recorded if the reference string 'gt_text' is equal to
  # 'pred_text'.
  exact_match_accuracy = sum(
      pred_text == gt_text
      for pred_text, gt_text in zip(pred_texts, groundtruth_texts)
  ) / len(pred_texts)
  return {
      "exact_match": exact_match_accuracy,
      "reference_in_sample": reference_in_sample_accuracy,
  }


# Define the task using the iterator callable.
qa_task = custom_tasks.CustomTask(
    name="my_iterator_qa",
    dataset=my_custom_iterator,
    metric_fn=text_metrics,
    generation_config=config.GenerationConfig(
        temperature=0.0, max_new_tokens=32, stop_sequences=["\n\n"]
    ),
)

# Register the task so the CLI can resolve it by name.
custom_tasks.TaskRegistry.global_registry().register(qa_task)
