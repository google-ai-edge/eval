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

"""Evaluation Pipeline orchestration logic."""

import importlib.resources
import json
import pathlib
import sys
import time
from typing import Any, cast

from model_eval.custom_tasks import groups
from model_eval.custom_tasks import registry as tasks_registry
from model_eval.frameworks import base as framework_base
from model_eval.frameworks import registry as framework_registry
from model_eval.runners import base as runner_base
from model_eval.runners import registry as runner_registry

import yaml

# Import core frameworks and runners to ensure they are registered by default.
import model_eval.frameworks.custom  # pylint: disable=unused-import,g-bad-import-order,g-import-not-at-top
import model_eval.frameworks.lm_eval  # pylint: disable=unused-import,g-bad-import-order,g-import-not-at-top
import model_eval.runners.litert_lm  # pylint: disable=unused-import,g-bad-import-order,g-import-not-at-top

# Registry of frameworks that support native library invocation.
_NATIVE_FRAMEWORKS = {
    framework_base.FrameworkType.LIGHTEVAL,
    framework_base.FrameworkType.LM_EVAL,
}


def _load_task_allowlist(
    path: str | None, framework: framework_base.FrameworkType
) -> list[str]:
  """Loads the allowlist of allowed tasks for a framework.

  Args:
      path: Path to a custom tasks.yaml file. If None, uses the bundled default.
      framework: The framework name (e.g. 'lm-eval') to look up tasks for.

  Returns:
      A list of allowed task names for the given framework.
  """
  if framework == framework_base.FrameworkType.CUSTOM:
    # Custom framework uses the registry's tasks and derived groups.
    names = tasks_registry.TaskRegistry.global_registry().get_all_tasks()
    return list(names) + groups.list_groups(names)
  if path:
    with open(path, "r", encoding="utf-8") as f:
      config = yaml.safe_load(f)
  else:
    # Use the bundled version.
    config_file = importlib.resources.files(
        "model_eval.config"
    ).joinpath("tasks.yaml")
    with importlib.resources.as_file(config_file) as f:
      with open(f, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

  return config.get(str(framework).replace("-", "_"), [])


def _expand_allowed_tasks(
    parents: list[str], framework: framework_base.FrameworkType
) -> set[str]:
  """Expands an allowlist so listing a parent implicitly authorizes its subtasks.

  Each parent stays in the result, and every concrete subtask reported by
  the framework's `subtasks_of` resolver is added too. This lets a user
  put just `mmlu` (lm-eval) or `bigbench_hard` (lighteval) in
  `tasks.yaml` and still pass validation when they request a leaf like
  `mmlu_abstract_algebra` or `bigbench_hard:causal_judgment`.

  Args:
      parents: The raw allowlist entries (parents and/or concrete tasks).
      framework: The framework whose `subtasks_of` resolver to consult.

  Returns:
      The union of the parents and every reachable subtask.
  """
  # Nothing to expand when there are no parents — and short-circuiting here
  # also avoids touching the framework registry, which lets callers with an
  # invalid framework value reach the downstream validation that diagnoses
  # that specific failure (e.g. `_validate` for NativeModelConfig).
  if not parents:
    return set()

  framework_impl = framework_registry.get_framework(framework)
  expanded: set[str] = set(parents)
  for parent in parents:
    expanded.update(framework_impl.subtasks_of(parent))
  return expanded


def _load_runner_allowlist(
    path: str | None, framework: framework_base.FrameworkType
) -> list[str]:
  """Loads the allowlist of allowed runners for a framework.

  Args:
      path: Path to a custom runners.yaml file. If None, uses the bundled
        default.
      framework: The framework name (e.g. 'lm-eval') to look up runners for.

  Returns:
      A list of allowed runner names for the given framework.
  """
  if path:
    with open(path, "r", encoding="utf-8") as f:
      config = yaml.safe_load(f)
  else:
    # Use the bundled version.
    config_file = importlib.resources.files(
        "model_eval.config"
    ).joinpath("runners.yaml")
    if config_file.is_file():
      with importlib.resources.as_file(config_file) as f:
        with open(f, "r", encoding="utf-8") as fh:
          config = yaml.safe_load(fh)
    else:
      config = {}

  return config.get(str(framework).replace("-", "_"), [])


class EvalPipeline:
  """Orchestrates model evaluation pipelines."""

  def __init__(
      self,
      model: runner_base.RunnerConfig | framework_base.NativeModelConfig,
      framework: framework_base.FrameworkType = framework_base.FrameworkType.LM_EVAL,
      tasks: tuple[str, ...] = (),
      limit: int | float | None = None,
      sample_range: tuple[int, int] | None = None,
      batch_size: int | None = None,
      output_dir: str = "results/",
      task_config: str | None = None,
      runner_config: str | None = None,
      eval_args: dict[str, Any] | None = None,
  ):
    """Initializes the evaluation pipeline orchestration logic.

    Args:
        model: The validated runner configuration or native model configuration
          object.
        framework: The target evaluation framework to trigger (e.g., 'lm-eval').
        tasks: Specific benchmark task names to evaluate on.
        limit: Maximum samples or fraction per task.
        sample_range: Range of samples to evaluate, space-separated (e.g., (10,
          20)).
        batch_size: Evaluation batch size.
        output_dir: Directory path to persist metrics and samples.
        task_config: Optional path to a custom tasks.yaml allowlist.
        runner_config: Optional path to a custom runners.yaml allowlist.
        eval_args: Extra evaluation framework arguments.
    """
    self._model = model
    self._framework = framework
    allowed_tasks = _load_task_allowlist(task_config, framework)
    # Implicitly authorize every subtask reachable from an allowed parent so
    # that e.g. listing `mmlu` in tasks.yaml lets a user request the leaf
    # `mmlu_abstract_algebra` without enumerating each subtask.
    expanded_allowed = _expand_allowed_tasks(allowed_tasks, framework)
    unknown = set(tasks) - expanded_allowed
    if unknown:
      raise ValueError(
          f"Tasks not in allowlist: {sorted(unknown)}. Pass task_config= to"
          " override."
      )

    allowed_runners = _load_runner_allowlist(runner_config, framework)
    # Resolve the identifier: use runner_type for custom runners, or model for
    # native.
    runner = getattr(model, "runner_type", getattr(model, "model", "native"))
    if allowed_runners and runner not in allowed_runners:
      raise ValueError(
          f"Runner not in allowlist for framework '{framework}': {runner}"
      )

    self._tasks = list(tasks)
    self._limit = limit
    self._sample_range = sample_range
    self._batch_size = batch_size
    self._output_dir = pathlib.Path(output_dir)
    self._output_dir.mkdir(parents=True, exist_ok=True)

    self._eval_args = dict(eval_args or {})
    self._validate()

  def _validate(self) -> None:
    """Validates pipeline constraints against native and custom runner execution paths."""
    if self._limit is not None and self._sample_range is not None:
      raise ValueError(
          "Conflicting options: 'limit' and 'sample_range' cannot be used"
          " together."
      )
    if isinstance(self._model, framework_base.NativeModelConfig):
      if self._framework == framework_base.FrameworkType.CUSTOM:
        raise ValueError(
            "framework='custom' requires a custom runner config, not a native"
            " model config."
        )
      if self._framework not in _NATIVE_FRAMEWORKS:
        raise ValueError(
            f"Framework '{self._framework}' does not support native model"
            " configs."
        )

  def run(self) -> framework_base.EvalResults:
    """Executes the evaluation workflow end-to-end via the selected route."""
    if isinstance(self._model, runner_base.RunnerConfig):
      # Custom runner-based execution path.
      return self._run_with_runner()
    else:
      # Native library execution path.
      return self._run_native()

  def _write_results(self, results: framework_base.EvalResults) -> None:
    """Persists metrics and any collected samples into JSON lines files across tasks."""
    ts = int(time.time())
    metrics_path = self._output_dir / f"run_{ts}_metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as f:
      for task_name, metrics in results.aggregated_metrics.items():
        for metric_name, value in metrics.items():
          f.write(
              json.dumps(
                  {"task": task_name, "metric": metric_name, "value": value}
              )
              + "\n"
          )

    if results.per_sample_outputs:
      samples_path = self._output_dir / f"run_{ts}_samples.jsonl"
      if self._sample_range:
        start, end = self._sample_range
        range_str = f"{end - start + 1} samples (index range: {start} to {end})"
        offset = start
      elif self._limit:
        range_str = (
            f"{self._limit} samples"
            if isinstance(self._limit, int)
            else f"{self._limit*100:.2f}% of samples"
        )
        offset = 0
      else:
        range_str = "all samples"
        offset = 0
      sys.stdout.write(f"\nWriting {range_str} per task to disk...\n")
      with open(samples_path, "w", encoding="utf-8") as f:
        for task_name, samples in results.per_sample_outputs.items():
          for idx, sample in enumerate(samples):
            sample_out = {
                "task": task_name,
                "sample_index": offset + idx,
                **sample,
            }
            f.write(json.dumps(sample_out) + "\n")
      print(f"Saved samples to {samples_path}")

    print(f"Saved metrics to {metrics_path}")

  def _run_with_runner(self) -> framework_base.EvalResults:
    """Triggers the execution via a custom runner instance."""
    model = cast(runner_base.RunnerConfig, self._model)
    runner_type = runner_base.RunnerType(model.runner_type)
    runner = runner_registry.create_runner(runner_type, model.model_dump())
    framework = framework_registry.get_framework(self._framework)
    with runner:
      results = framework.evaluate(
          runner=runner,
          tasks=self._tasks,
          limit=self._limit,
          sample_range=self._sample_range,
          batch_size=self._batch_size,
          eval_args=self._eval_args,
      )
    self._write_results(results)
    return results

  def _run_native(self) -> framework_base.EvalResults:
    """Triggers the native evaluation loop using the direct library fallback path."""
    model = cast(framework_base.NativeModelConfig, self._model)
    framework = framework_registry.get_framework(self._framework)
    results = framework.evaluate_native(
        model_config=model,
        tasks=self._tasks,
        limit=self._limit,
        sample_range=self._sample_range,
        batch_size=self._batch_size,
        eval_args=self._eval_args,
    )
    self._write_results(results)
    return results

  @classmethod
  def _format_fields(cls, fields: list[dict[str, Any]], header: str) -> str:
    """Formats a list of argument fields into a printable string with column names.

    Args:
        fields: A list of dictionaries, each containing 'name', 'type', and
          'default' keys.
        header: A header string to prepend to the formatted fields.

    Returns:
        A formatted string representation of the argument fields.
    """
    if not fields:
      return f"\n{header}"

    # Define column titles.
    col_n, col_t, col_d = "Argument", "Type", "Default"

    # Calculate the maximum width, including the column titles themselves.
    max_n = max([len(str(f.get("name", ""))) for f in fields] + [len(col_n)])
    max_t = max([len(str(f.get("type", ""))) for f in fields] + [len(col_t)])

    res = [f"\n{header}"]
    # Add column names row.
    res.append(f"  {col_n:<{max_n}}  {col_t:<{max_t}}  {col_d}")
    # Add a separator line for visual clarity.
    res.append(f"  {'-' * max_n}  {'-' * max_t}  {'-' * len(col_d)}")

    for f in fields:
      n, t, d = f.get("name", ""), f.get("type", ""), f.get("default", "")
      # Use dynamic padding based on the longest field value or column title.
      res.append(f"  {n:<{max_n}}  {t:<{max_t}}  {d}")
    return "\n".join(res)

  @classmethod
  def describe_runner_args(cls, runner: str) -> str:
    """Retrieves and formats the runtime arguments supported by a given model runner.

    Args:
        runner: The string identifier of the runner type.

    Returns:
        A formatted description of the runner's configurable arguments, or a
        string indicating the runner is unknown.
    """
    # Attempt to resolve the identifier against registered custom runners.
    # If found, query the custom runner for its argument fields.
    try:
      runner_type = runner_base.RunnerType(runner)
      runner_cls = runner_registry.get_runner_cls(runner_type)
      if hasattr(runner_cls, "describe_runner_args"):
        return cls._format_fields(
            runner_cls.describe_runner_args(), f"Runner args for: {runner}"
        )
    except (ValueError, KeyError):
      pass

    # Fallback: Query registered frameworks to check if they natively support
    # the runner.
    # If found, query the native runner for its argument fields.
    for fw_type in framework_base.FrameworkType:
      fw = framework_registry.get_framework(fw_type)
      try:
        return cls._format_fields(
            fw.describe_native_runner_args(runner),
            f"Runner args for: {runner} [via {fw_type}]",
        )
      except NotImplementedError:
        continue

    # Identifier not matched in either custom runners or native frameworks.
    return f"Unknown runner: '{runner}'"

  @classmethod
  def describe_eval_args(cls, framework: str) -> str:
    """Retrieves and formats the evaluation arguments supported by a given framework.

    Args:
        framework: The string identifier of the evaluation framework.

    Returns:
        A formatted description of the framework's evaluation arguments, or a
        string indicating the framework is unknown.
    """
    try:
      fw_type = framework_base.FrameworkType(framework)
      # Retrieve the registered framework instance.
      fw = framework_registry.get_framework(fw_type)
      # If found, extract and format the evaluation argument fields.
      return cls._format_fields(
          fw.describe_eval_args(), f"Eval args for: {framework}"
      )
    except (ValueError, KeyError):
      return f"Unknown framework: '{framework}'"
