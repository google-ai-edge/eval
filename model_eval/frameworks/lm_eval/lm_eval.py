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

"""lm-eval framework evaluation adapter for ai_edge_eval."""

from typing import Any
import warnings

from model_eval.api import constants
from model_eval.frameworks import base
from model_eval.frameworks import registry
from model_eval.runners import base as runners_base
from model_eval.utils import introspection
import lm_eval
from lm_eval import tasks as lm_eval_tasks

from model_eval.frameworks import _local_chat_score_model  # pylint: disable=unused-import,g-bad-import-order


def _model_args(runner: runners_base.AbstractRunner) -> str:
  """Constructs the model arguments for generation/chat tasks.

  Args:
      runner: The runner instance to construct arguments for.

  Returns:
      A string containing the formatted model arguments.
  """
  return f"base_url={runner.server_url},model={runner.model_name}"


def _parse_lm_eval_results(raw: dict[str, Any] | None) -> base.EvalResults:
  """Extracts the results from task evaluations."""
  if not raw:
    return base.EvalResults(
        framework_type="lm-eval",
        aggregated_metrics={},
        per_sample_outputs={},
        metadata={},
    )

  _task_keys = ("results", "samples", "versions", "n-shot", "higher_is_better")
  parsed = {k: {} for k in _task_keys}

  for key in _task_keys:
    if key in raw and isinstance(raw[key], dict):
      parsed[key].update(raw[key])

  # Gather metadata.
  metadata = {}
  for meta_key in ("versions", "n-shot", "higher_is_better"):
    if meta_key in parsed:
      metadata[meta_key] = parsed[meta_key]

  if "config" in raw:
    metadata["config"] = raw["config"]

  if "lm_eval_version" in raw:
    metadata["lm_eval_version"] = raw["lm_eval_version"]

  return base.EvalResults(
      framework_type="lm-eval",
      aggregated_metrics=parsed["results"],
      per_sample_outputs=parsed["samples"],
      metadata=metadata,
  )


@registry.register_framework(base.FrameworkType.LM_EVAL)
class LmEvalFramework(base.AbstractEvalFramework):
  """Evaluates models using EleutherAI's lm-eval harness."""

  def evaluate(
      self,
      runner: runners_base.AbstractRunner,
      tasks: list[str],
      limit: int | float | None = None,
      batch_size: int | None = None,
      eval_args: dict[str, Any] | None = None,
  ) -> base.EvalResults:
    """Evaluates the model using the provided runner.

    Args:
        runner: The runner instance to evaluate against.
        tasks: A list of task names to evaluate.
        limit: Limit for samples.
        batch_size: Evaluation batch size.
        eval_args: Additional evaluation arguments.

    Returns:
        The results of the evaluation.
    """
    params = self._from_unified_eval_args(
        limit, batch_size, eval_args, default_batch_size=1
    )

    num_fewshot = params.eval_args.pop("num_fewshot", 0)
    # By default, apply_chat_template is set to True for custom runner as the
    # custom runner applies chat templates internally.
    apply_chat_template = params.eval_args.pop("apply_chat_template", True)
    if not apply_chat_template:
      raise ValueError(
          "apply_chat_template must be True for "
          f"{constants.LOCAL_CHAT_SCORE_MODEL_NAME} based evaluation."
      )
    task_manager = lm_eval_tasks.TaskManager()

    # Emits a warning for tasks that depend on is_greedy if the underlying
    # server is configured with the always_return_not_greedy=True optimization.
    if hasattr(runner, "returns_greedy") and not runner.returns_greedy:
      self._warn_loglikelihood_tasks(tasks, task_manager)

    with runner:
      raw_results = lm_eval.simple_evaluate(
          model=constants.LOCAL_CHAT_SCORE_MODEL_NAME,
          model_args=_model_args(runner),
          tasks=tasks,
          num_fewshot=num_fewshot,
          batch_size=params.batch_size,
          limit=params.limit,
          task_manager=task_manager,
          apply_chat_template=apply_chat_template,
          **params.eval_args,
      )

    return _parse_lm_eval_results(raw_results)

  def evaluate_native(
      self,
      model_config: base.NativeModelConfig,
      tasks: list[str],
      limit: int | float | None = None,
      batch_size: int | None = None,
      eval_args: dict[str, Any] | None = None,
  ) -> base.EvalResults:
    """Evaluates a model natively using lm-eval without starting a server runner.

    Args:
        model_config: The native model configuration.
        tasks: A list of task names to evaluate.
        limit: Limit for samples.
        batch_size: Evaluation batch size.
        eval_args: Additional evaluation arguments.

    Returns:
        The results of the evaluation.
    """
    eval_params = self._from_unified_eval_args(limit, batch_size, eval_args)

    num_fewshot = eval_params.eval_args.pop("num_fewshot", 0)
    # By default, apply_chat_template is set to True to avoid unexpected
    # inconsistencies between native and custom runners.
    apply_chat_template = eval_params.eval_args.pop("apply_chat_template", True)

    task_manager = lm_eval_tasks.TaskManager()

    # Dynamically determine the model path argument key for native runners.
    model_path_key = self._get_model_path_key(model_config.model)
    # By default, device_key is always set to 'device' for native runners.
    device_key = "device"
    # Resolve the runner args for the native runner, avoiding conflicts between
    # unified CLI flags and runner_args.
    runner_args = self._from_unified_runner_args(
        model_config, model_path_key=model_path_key, device_key=device_key
    )

    # Pop device key from runner_args to avoid conflicts with the device flag in
    # lm_eval.simple_evaluate().
    runner_args.model_args.pop(device_key, None)
    raw_results = lm_eval.simple_evaluate(
        model=model_config.model,
        model_args=runner_args.model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=eval_params.batch_size,
        device=runner_args.device,
        limit=eval_params.limit,
        task_manager=task_manager,
        apply_chat_template=apply_chat_template,
        **eval_params.eval_args,
    )

    return _parse_lm_eval_results(raw_results)

  @classmethod
  def supported_task_ids(cls) -> list[str]:
    """Returns a list of all raw task IDs from lm-eval."""
    return list(lm_eval_tasks.TaskManager().all_tasks)

  @classmethod
  def supported_runners(cls) -> list[str]:
    """Returns a list of custom runners and native runners supported by the framework."""
    from model_eval.runners import registry as runner_registry  # pylint: disable=g-import-not-at-top
    from lm_eval.api import registry as lm_eval_registry  # pylint: disable=g-import-not-at-top
    # Ensure lm-eval natively supported models are registered into the registry.
    import lm_eval.models as lm_eval_models  # pylint: disable=unused-import,g-import-not-at-top

    custom_runners = runner_registry.get_all_runners()
    native_runners = list(lm_eval_registry.MODEL_REGISTRY.keys())
    # Merge and sort the native and custom runners.
    return sorted(list(set(custom_runners + native_runners)))

  def _warn_loglikelihood_tasks(
      self, tasks: list[str], task_manager: lm_eval_tasks.TaskManager
  ) -> None:
    """Detects loglikelihood tasks and emits instructions warning users about zero accuracy when server-side greedy checking is suppressed.

    Args:
        tasks: The list of task names currently being evaluated via lm-eval.
        task_manager: The task manager to use for loading tasks.
    """
    # Filter the list of tasks to find those explicitly relying on the
    # 'loglikelihood' output type (which computes output accuracy directly from
    # the underlying server's is_greedy status). Note that `multiple_choice`
    # scoring tasks are not affected.
    affected = [
        name
        for name in tasks
        if any(
            getattr(t, "OUTPUT_TYPE", None) == "loglikelihood"
            for t in task_manager.load_task_or_group([name]).values()
        )
    ]
    # Emit the warning only if any affected tasks are detected.
    if affected:
      warnings.warn(
          f"Tasks {affected} use output_type='loglikelihood' and compute acc"
          " from is_greedy. The server is configured with"
          " always_return_not_greedy=True — acc will be 0 for these tasks. Use"
          " --model-type hf (native path) to evaluate them correctly.",
          stacklevel=2,
      )

  @classmethod
  def describe_eval_args(cls) -> list[dict[str, Any]]:
    return introspection.get_fields(lm_eval.simple_evaluate)

  @classmethod
  def describe_native_runner_args(cls, runner: str) -> list[dict[str, Any]]:
    from lm_eval.api import registry as lm_eval_registry  # pylint: disable=g-import-not-at-top

    if runner in lm_eval_registry.MODEL_REGISTRY:
      # Retrieve the given runner name from the lm-eval registry.
      # If found, query the native runner for its argument fields.
      return introspection.get_fields(lm_eval_registry.get_model(runner))
    # Identifier not matched in the lm-eval registry.
    raise NotImplementedError

  def _get_model_path_key(self, model: str) -> str:
    """Dynamically retrieves the model path argument key for a native model.

    Inspects the constructor signature of the specified lm-eval model wrapper.

    Args:
        model: The model registry name (e.g., 'hf').

    Raises:
        ValueError: If the signature cannot be retrieved from the model wrapper.

    Returns:
        The parameter key corresponding to the model path/weights ('pretrained'
        or 'model'). Defaults to 'pretrained'.
    """
    import inspect  # pylint: disable=g-import-not-at-top
    from lm_eval.api import registry as lm_eval_registry  # pylint: disable=g-import-not-at-top

    try:
      sig = inspect.signature(lm_eval_registry.get_model(model).__init__)
    except Exception as e:  # pylint: disable=broad-except
      raise ValueError(f"Failed to get signature for model {model}: {e}") from e
    if "pretrained" in sig.parameters:
      return "pretrained"
    elif "model" in sig.parameters:
      return "model"
    return "pretrained"
