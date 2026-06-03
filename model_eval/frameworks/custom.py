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

"""Custom generation-only framework implementation."""

import math
from typing import Any

from model_eval import config
from model_eval.api import constants as api_constants
from model_eval.custom_tasks import base as tasks_base
from model_eval.custom_tasks import loaders
from model_eval.custom_tasks import registry as tasks_registry
from model_eval.frameworks import base
from model_eval.frameworks import registry
from model_eval.runners import base as runners_base
from model_eval.utils import introspection
import httpx
import tqdm


def _apply_limit(
    rows: list[dict[str, Any]], limit: int | float
) -> list[dict[str, Any]]:
  """Slices a dataset using an absolute count limit or a percentage fraction.

  Args:
      rows: The input list of dataset rows to sample from.
      limit: Absolute integer count or a float representing a fraction.

  Returns:
      The sliced list of rows.
  """
  n = (
      int(math.ceil(len(rows) * limit))
      if isinstance(limit, float) and limit < 1
      else int(limit)
  )
  return rows[:n]


def _apply_samples(
    rows: list[dict[str, Any]], samples_val: Any, task_name: str
) -> list[dict[str, Any]]:
  """Filters a dataset by keeping precise row indices or resolving range slices.

  Args:
      rows: The list of dataset rows to slice from.
      samples_val: A dictionary mapping task names to indices/ranges, a list of
        indices, or a slice expression string.
      task_name: The specific name of the current task being evaluated.

  Raises:
      ValueError: If samples_val is not a dict, str, or list.

  Returns:
      The filtered dataset rows.
  """

  if not isinstance(samples_val, (dict, str, list)):
    raise ValueError(
        "Invalid type for samples: expected dict, str, or list, got"
        f" {type(samples_val).__name__}"
    )

  if isinstance(samples_val, dict):
    if task_name in samples_val:
      task_expr = samples_val[task_name]
    else:
      return rows
  else:
    task_expr = samples_val

  if isinstance(task_expr, list):
    indices = [i for i in task_expr if i < len(rows)]
  else:
    indices = loaders.parse_samples(str(task_expr), len(rows))
  return [rows[i] for i in indices]


@registry.register_framework("custom")
class CustomFramework(base.AbstractEvalFramework):
  """Framework driving generation-only custom tasks against any server loop."""

  def evaluate(
      self,
      runner: runners_base.AbstractRunner,
      tasks: list[str],
      limit: int | float | None = None,
      sample_range: tuple[int, int] | None = None,
      batch_size: int | None = None,
      eval_args: dict[str, Any] | None = None,
  ) -> base.EvalResults:
    """Evaluates the runner across the specified custom tasks.

    Args:
      runner: The target runner implementation responsible for server inference.
      tasks: A list of unique task names to look up and evaluate.
      limit: Optional limit on number of samples per task to generate.
      sample_range: Optional range of samples to evaluate.
      batch_size: Evaluation batch size, used for conflict checks/parity.
      eval_args: Additional evaluation configurations.

    Raises:
      ValueError: If conflicting limit/sample options are provided.

    Returns:
      An EvalResults instance containing all aggregated task metrics and
      outputs. The per-sample outputs contain the keys 'input', 'prediction',
      and 'ground_truth'.
    """
    eval_params = self._from_unified_eval_args(
        limit, sample_range, batch_size, eval_args
    )

    samples = eval_params.eval_args.pop("samples", None)
    if eval_params.limit is not None and eval_params.sample_range is not None:
      raise ValueError(
          "Only one of 'limit' or 'sample_range' can be set, not both."
      )
    if eval_params.sample_range is not None and samples is not None:
      raise ValueError(
          "Only one of 'sample_range' or 'samples' can be set, not both."
      )
    if eval_params.limit is not None and samples is not None:
      raise ValueError("Only one of 'limit' or 'samples' can be set, not both.")

    reg = tasks_registry.TaskRegistry.global_registry()
    all_results = {}
    all_samples = {}
    with httpx.Client(timeout=120.0) as http_client:
      for name in tasks:
        task = reg.get_task(name)
        all_results[name], all_samples[name] = self._run_task(
            runner,
            task,
            http_client,
            limit=eval_params.limit,
            sample_range=eval_params.sample_range,
            samples=samples,
        )
    return base.EvalResults(
        framework_type="custom",
        aggregated_metrics=all_results,
        per_sample_outputs=all_samples,
        metadata={},
    )

  def _run_task(
      self,
      runner: runners_base.AbstractRunner,
      task: tasks_base.CustomTask,
      http_client: httpx.Client,
      *,
      limit: int | float | None = None,
      sample_range: tuple[int, int] | None = None,
      samples: Any = None,
  ) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Executes a single custom evaluation task.

    Loads the dataset, applies any slicing bounds, drives generation for each
    row, calculates user-defined metrics, and maps inputs to predictions.

    Args:
      runner: The target runner supporting HTTP completions inference.
      task: The CustomTask instance defining the dataset and metric hooks.
      http_client: Explicit httpx client instance to use for API calls.
      limit: Optional limit on number of samples per task to generate.
      sample_range: Optional range of samples to evaluate.
      samples: Optional explicit sample indices or dictionary map.

    Raises:
      ValueError: If conflicting limit/sample options are provided.

    Returns:
      A tuple containing the dict of aggregated metrics and a list of per-sample
      execution dictionaries. Each dictionary contains the keys: 'input' (the
      input messages), 'prediction' (the generated text), and 'ground_truth'
      (the expected output).
    """
    if limit is not None and sample_range is not None:
      raise ValueError(
          "Only one of 'limit' or 'sample_range' can be set, not both."
      )
    if sample_range is not None and samples is not None:
      raise ValueError(
          "Only one of 'sample_range' or 'samples' can be set, not both."
      )
    if limit is not None and samples is not None:
      raise ValueError("Only one of 'limit' or 'samples' can be set, not both.")

    if sample_range is not None:
      samples = f"{sample_range[0]}-{sample_range[1]}"

    # Load entire dataset lazily into an initial list buffer.
    rows: list[tasks_base.DatasetRow] = list(loaders.load_dataset(task.dataset))

    if limit:
      rows = _apply_limit(rows, limit)

    # Apply explicit 'samples' string expression (e.g.,
    # {"task_name_1": "0-10", "task_name_2": "1,3,5"}) filtering.
    if samples:
      rows = _apply_samples(rows, samples, task.name)

    predictions = []
    groundtruths = []
    # sample_outputs: List of dicts mapping user prompt, generated text, and
    # target ground truth for per-sample debugging.
    sample_outputs: list[dict[str, Any]] = []

    # Iterate and generate model completions row by row.
    for row in tqdm.tqdm(rows, desc=f"Evaluating {task.name}"):
      input_msgs = row["messages"]
      gt = row["ground_truth"]

      # Generate the prediction.
      pred = self._generate(
          runner, input_msgs, task.generation_config, http_client
      )

      # Populate the lists required for metric calculation and output logging.
      predictions.append(pred)
      groundtruths.append(gt)
      sample_outputs.append({
          "input": input_msgs,
          "prediction": pred,
          "ground_truth": gt,
      })

    result = task.metric_fn(iter(predictions), iter(groundtruths), iter(rows))
    return result, sample_outputs

  def _generate(
      self,
      runner: runners_base.AbstractRunner,
      input_messages: tasks_base.OpenAIMessages,
      generation_config: config.GenerationConfig,
      http_client: httpx.Client,
  ) -> tasks_base.PredictionType:
    """Generates a model prediction for a single input row.

    Submits a standard chat completion payload and extracts the resulting turn.

    Args:
      runner: The runner instance exposing the base endpoint.
      input_messages: The input messages context for generation.
      generation_config: The configuration parameters for generation.
      http_client: Explicit httpx client instance to use for API calls.

    Returns:
      The generated prediction.
    """
    # Transmit the isolated chat messages context to the OpenAI-compatible
    # endpoint.
    resp = http_client.post(
        f"{runner.server_url}/{api_constants.CHAT_COMPLETIONS_ENDPOINT}",
        json={
            "model": runner.model_name,
            "messages": input_messages,
            "temperature": generation_config.temperature,
            "max_tokens": generation_config.max_new_tokens,
            "stop": generation_config.stop_sequences or None,
        },
    )
    resp.raise_for_status()
    # We currently request only a single completion per generation call,
    # so we can safely extract the text from the first and only choice.
    return resp.json()["choices"][0]["message"]["content"]

  @classmethod
  def supported_task_ids(cls) -> list[str]:
    reg = tasks_registry.TaskRegistry.global_registry()
    return reg.get_all_tasks()

  @classmethod
  def supported_runners(cls) -> list[str]:
    """Returns a list of all registered custom runners supported by the custom framework."""
    from model_eval.runners import registry as runner_registry  # pylint: disable=g-import-not-at-top

    return runner_registry.get_all_runners()

  @classmethod
  def describe_eval_args(cls) -> list[dict[str, Any]]:
    return introspection.get_fields(cls.evaluate)
