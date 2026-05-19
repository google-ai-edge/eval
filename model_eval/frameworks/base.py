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

"""Base framework abstractions."""

import abc
import dataclasses
import enum
from typing import Any
from typing import TYPE_CHECKING

import pydantic

if TYPE_CHECKING:
  from model_eval.runners import base as runners_base  # pylint: disable=g-bad-import-order


class FrameworkType(enum.StrEnum):
  """Supported evaluation framework types."""

  LM_EVAL = "lm-eval"
  CUSTOM = "custom"


class EvalResults(pydantic.BaseModel):
  """Aggregated result from an evaluation framework run."""

  framework_type: str
  # Aggregated metrics per task.
  # Keys are task names, and values are dicts mapping metric names to values.
  aggregated_metrics: dict[str, dict[str, Any]]
  # Per-sample outputs per task.
  # Keys are task names, and values are a list of dicts with the details of each
  # sample.
  per_sample_outputs: dict[str, list[dict[str, Any]]] = {}
  # Metadata including versions, n-shot, higher_is_better, configs, etc.
  metadata: dict[str, Any] = {}


class NativeModelConfig(pydantic.BaseModel):
  """Native model config used across frameworks."""

  # Model name as the target framework understands it.
  model: str
  # Unified path identifier for the model.
  model_path: str | None = None
  # Unified target execution device/backend.
  device: str | None = None
  # Framework-specific keyword arguments dictionary.
  model_args: dict[str, Any] = {}


@dataclasses.dataclass
class ResolvedEvalParams:
  """Container for resolved evaluation parameters."""

  limit: int | float | None
  batch_size: int | None
  eval_args: dict[str, Any]


@dataclasses.dataclass
class ResolvedRunnerArgs:
  """Container for resolved runner parameters in native mode."""

  model_path: str | None
  device: str | None
  model_args: dict[str, Any]


class AbstractEvalFramework(abc.ABC):
  """Base interface for all evaluation frameworks."""

  @abc.abstractmethod
  def evaluate(
      self,
      runner: "runners_base.AbstractRunner",
      tasks: list[str],
      limit: int | float | None = None,
      batch_size: int | None = None,
      eval_args: dict[str, Any] | None = None,
  ) -> EvalResults:
    """Evaluates a runner against a list of tasks.

    Args:
        runner: The runner implementation to evaluate.
        tasks: An iterable of task names to evaluate.
        limit: Maximum samples or fraction per task.
        batch_size: Evaluation batch size.
        eval_args: Extra evaluation arguments.

    Returns:
        The EvalResults instance containing the evaluation results.
    """
    ...

  def evaluate_native(
      self,
      model_config: NativeModelConfig,
      tasks: list[str],
      limit: int | float | None = None,
      batch_size: int | None = None,
      eval_args: dict[str, Any] | None = None,
  ) -> EvalResults:
    """Evaluates the native model configuration against the provided tasks.

    Args:
        model_config: The native model configuration to evaluate.
        tasks: An iterable of task names to evaluate.
        limit: Maximum samples or fraction per task.
        batch_size: Evaluation batch size.
        eval_args: Extra evaluation arguments.

    Returns:
        The EvalResults instance containing the evaluation results.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not support native model configs."
    )

  def _consume_arg(
      self,
      explicit_val: Any,
      args_dict: dict[str, Any],
      dict_key: str,
      flag_name: str,
      source_name: str,
      default: Any = None,
  ) -> Any:
    """Validates no framework argument conflict exists and returns the effective value.

    Checks whether a parameter is redundantly specified both via an explicit
    unified flag and a dynamic keyword argument, raising a clear error if so.
    If no explicit value is given, extracts the relevant key from the argument
    dictionary or falls back to a default.

    Args:
        explicit_val: The value passed explicitly from a unified parameter flag.
        args_dict: The dictionary of dynamic framework/runner parameters.
        dict_key: The dynamic dictionary key corresponding to this argument.
        flag_name: The name of the unified CLI flag (e.g., '--limit').
        source_name: The dictionary source flag name (e.g., '--eval-args').
        default: A fallback default value if the key is absent.

    Raises:
        ValueError: If both the explicit flag and the dynamic key are specified.

    Returns:
        The resolved parameter value.
    """
    if explicit_val is not None:
      if dict_key in args_dict:
        raise ValueError(
            f"{flag_name} conflicts with '{dict_key}' in {source_name}. "
            "Use one or the other."
        )
      return explicit_val
    return args_dict.pop(dict_key, default)

  def _from_unified_eval_args(
      self,
      limit: int | float | None,
      batch_size: int | None,
      eval_args: dict[str, Any] | None,
      limit_key: str = "limit",
      batch_size_key: str = "batch_size",
      default_batch_size: int | None = None,
  ) -> ResolvedEvalParams:
    """Resolves general evaluation constraints from unified flags and parameter overrides.

    Safely resolves limits and batch sizing settings, translating unified flag
    expectations into target framework-specific fields without collisions.

    Args:
        limit: Sample count or evaluation fraction limit.
        batch_size: Evaluation batch size.
        eval_args: Additional evaluation framework arguments dictionary.
        limit_key: Underlying framework key for the limit arg.
        batch_size_key: Underlying framework key for the batch size arg.
        default_batch_size: Default batch size if absent in both sources.

    Returns:
        Resolved parameter mapping container holding the limit, batch size, and
        remaining evaluation argument overrides.
    """
    args = dict(eval_args or {})
    res_limit = self._consume_arg(
        limit, args, limit_key, "--limit", "--eval-args"
    )
    res_batch = self._consume_arg(
        batch_size,
        args,
        batch_size_key,
        "--batch-size",
        "--eval-args",
        default=default_batch_size,
    )
    return ResolvedEvalParams(
        limit=res_limit, batch_size=res_batch, eval_args=args
    )

  def _from_unified_runner_args(
      self,
      model_config: NativeModelConfig,
      model_path_key: str = "model_path",
      device_key: str = "device",
  ) -> ResolvedRunnerArgs:
    """Resolves native execution parameters against dynamic runner arguments.

    Parses model weights path and target inference device mapping securely to
    avoid overriding explicit top-level parameters via dynamic configurations.

    Args:
        model_config: The native framework model configuration settings.
        model_path_key: Underlying framework parameter for the model weights
          path.
        device_key: Underlying framework parameter for targeted inference
          device.

    Returns:
        Resolved underlying model parameters mapping container.
    """
    model_args = dict(model_config.model_args)
    res_path = self._consume_arg(
        model_config.model_path,
        model_args,
        model_path_key,
        "--model-path",
        "--runner-args",
    )
    res_device = self._consume_arg(
        model_config.device, model_args, device_key, "--device", "--runner-args"
    )
    # Merge resolved values into the model_args dictionary for frameworks that
    # use the model_args dictionary for runner configuration.
    if res_path is not None:
      model_args[model_path_key] = res_path
    if res_device is not None:
      model_args[device_key] = res_device
    return ResolvedRunnerArgs(
        model_path=res_path, device=res_device, model_args=model_args
    )

  @classmethod
  @abc.abstractmethod
  def supported_task_ids(cls) -> list[str]:
    """Returns a list of all supported task identifier strings.

    Returns:
        A list of task names natively supported by the framework.
    """
    ...

  @classmethod
  @abc.abstractmethod
  def supported_runners(cls) -> list[str]:
    """Returns a list of all supported runner strings.

    Returns:
        A list of runners natively supported by the framework.
    """
    ...

  @classmethod
  @abc.abstractmethod
  def describe_eval_args(cls) -> list[dict[str, Any]]:
    """Returns descriptions of accepted evaluation arguments."""
    ...

  @classmethod
  def describe_native_runner_args(cls, runner: str) -> list[dict[str, Any]]:
    """Returns native runner argument descriptions."""
    raise NotImplementedError
