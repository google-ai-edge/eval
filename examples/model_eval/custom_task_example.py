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
  --custom-tasks-file examples/model_eval/custom_task_example.py \
  --tasks my_iterator_qa \
  --output-dir /tmp/results
```

Against an already running OpenAI-compatible server, name the model the server
should use via `model_name` in `--runner-args`:
```bash
ai-edge-eval \
  --runner http-server \
  --runner-args "server_url=http://<address>:<port>,model_name=<model_name>" \
  --framework custom \
  --eval-args "samples={'my_iterator_qa':'1-3'}" \
  --custom-tasks-file examples/model_eval/custom_task_example.py \
  --tasks my_iterator_qa \
  --output-dir /tmp/results
```
"""

from typing import Any, Iterator

from model_eval import config
from model_eval import custom_tasks

_GENERATION_CONFIG = config.GenerationConfig(
    temperature=0.0, max_new_tokens=32, stop_sequences=["\n\n"]
)


def my_custom_iterator(
    task_args: custom_tasks.TaskArgs | None = None,
) -> Iterator[custom_tasks.DatasetRow]:
  """Generator that yields DatasetRow (input + ground truth)."""
  # `model_name` is provided by the runner's server_args (e.g. `--runner-args
  # "...,model_name=..."` for http-server); unset for local runners, in which
  # case `chat_request` falls back to the default.
  model_name = (task_args or {}).get("model_name")
  dataset = [
      {"q": "What is the capital of France?", "a": "Paris"},
      {"q": "What is 2+2?", "a": "4"},
      {"q": "Who wrote Hamlet?", "a": "Shakespeare"},
      {"q": "What is the color of Mars?", "a": "Red"},
  ]
  for row in dataset:
    yield {
        "requests": [
            custom_tasks.chat_request(
                [{"role": "user", "content": row["q"]}],
                _GENERATION_CONFIG,
                model_name=model_name,
            )
        ],
        "ground_truth": row["a"],
    }


def text_metrics(
    preds: Iterator[list[dict[str, Any] | None]],
    groundtruths: Iterator[str],
    rows: Iterator[custom_tasks.DatasetRow],
):
  """Evaluates if the ground truth is contained within the model's output."""
  del rows  # Unused.
  if not preds:
    return {"exact_match": 0.0, "reference_in_sample": 0.0}
  pred_texts = [
      (custom_tasks.chat_response_text(p[0]) or "").lower() if p else ""
      for p in preds
  ]
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
    endpoint="v1/chat/completions",
    dataset=my_custom_iterator,
    metric_fn=text_metrics,
)

# Register the task so the CLI can resolve it by name.
custom_tasks.TaskRegistry.global_registry().register(qa_task)
