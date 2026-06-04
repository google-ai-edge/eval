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

"""Adapter for the Lighteval evaluation framework."""

import dataclasses
import functools
import importlib
import types
from typing import Any
import warnings

from model_eval.frameworks import base
from model_eval.frameworks import registry
from model_eval.frameworks.lighteval import _chat_score_lighteval_model
from model_eval.runners import base as runners_base
from model_eval.utils import introspection
from lighteval import pipeline as lighteval_pipeline  # type: ignore
from lighteval.logging import evaluation_tracker  # type: ignore
from lighteval.models.endpoints import litellm_model  # type: ignore
from lighteval.tasks import lighteval_task  # type: ignore
from lighteval.tasks import registry as lighteval_registry  # type: ignore
import litellm


@dataclasses.dataclass(frozen=True)
class _NativeRunnerInfo:
  """Metadata for resolving a native lighteval model configuration.

  Attributes:
      module_path: The path to the module containing the lighteval model config
        class.
      config_class: The name of the lighteval model config class.
      model_path_field_name: The field name in the config class for specifying
        the model path or name.
      batch_size_field_name: The field name in the config class for specifying
        the batch size.
      device_field_name: The field name in the config class for specifying the
        device.
  """

  module_path: str
  config_class: str
  model_path_field_name: str | None
  batch_size_field_name: str | None
  device_field_name: str | None


# Mapping of native runner names to their corresponding lighteval model config
# classes, used for introspection to parse runner-specific arguments.
_NATIVE_RUNNER_CONFIGS = {
    "accelerate": _NativeRunnerInfo(
        module_path="lighteval.models.transformers.transformers_model",
        config_class="TransformersModelConfig",
        model_path_field_name="model_name",
        batch_size_field_name="batch_size",
        device_field_name="device",
    ),
}


def configure_lighteval_slicing(
    limit: int | float | None = None,
    sample_range: tuple[int, int] | None = None,
):
  """Configures LightevalTask.get_docs to support slicing data within [start, end].

  Args:
      limit: Maximum number of samples to evaluate per task if integer, or
        fraction of samples if float. Defaults to None.
      sample_range (list/tuple): e.g., [10, 20], meaning it fetches data from
        index 10 to 19.
  """
  # Safely backup the original method to prevent redundant patching.
  if not hasattr(lighteval_task.LightevalTask, "_original_get_docs"):
    lighteval_task.LightevalTask._original_get_docs = (
        lighteval_task.LightevalTask.get_docs
    )

  def slicing_get_docs(
      self_instance, max_samples: int | None = None, *args, **kwargs
  ):
    if sample_range:
      assert (
          len(sample_range) == 2
      ), "sample_range must be a list or tuple of two integers, e.g., [10, 20]."
      assert sample_range[0] <= sample_range[1], "start must be <= end."

      start = sample_range[0]
      end = sample_range[1]

      # Calls the original get_docs method.
      #
      # At this point, `max_samples` is already set to the right
      # boundary of our slice (`end`). The underlying Lighteval framework
      # eagerly loads the data and performs the following operations:
      #
      # 1. Fully loads the dataset and applies the task's prompt function to
      #    convert raw dataset items into `Doc` objects.
      # 2. Performs a global in-memory shuffle of the entire dataset using a
      #    fixed random seed (seed=42).
      # 3. Truncates the list to keep only the first `max_samples` items.
      # 4. Assembles few-shot examples and injects generation parameters for
      #    these items.
      #
      # It is guaranteed to return a standard Python list (`list[Doc]`)
      # containing the first `end` fully processed evaluation samples.
      docs = lighteval_task.LightevalTask._original_get_docs(
          self_instance, end + 1, *args, **kwargs
      )
      if start > len(docs):
        raise ValueError(
            f"Start index {start} exceeds dataset length {len(docs)}."
        )
      if end + 1 > len(docs):
        warnings.warn(
            f"End index {end} exceeds dataset length {len(docs)}. Adjusting end"
            f" to {len(docs) - 1}."
        )
        end = len(docs) - 1
      return docs[start : end + 1]
    elif limit:
      if isinstance(limit, float):
        # Calls the original get_docs method without a max_samples bound.
        #
        # Eagerly loads the full dataset to enable dynamic percentage sampling:
        #
        # 1. Fully loads the dataset and applies the task's prompt function to
        #    convert raw dataset items into `Doc` objects.
        # 2. Performs a global in-memory shuffle of the entire dataset using a
        #    fixed random seed (seed=42).
        # 3. Assembles few-shot examples and injects generation parameters for
        #    all items in the dataset.
        #
        # Once the fully processed list (`list[Doc]`) is returned, we calculate
        # the slice count based on the actual length (`int(len(docs) * limit)`)
        # and return the leading fraction of evaluation samples.
        docs = lighteval_task.LightevalTask._original_get_docs(
            self_instance, None, *args, **kwargs
        )
        end = max(int(len(docs) * limit), 1)
        return docs[:end]
      else:
        return lighteval_task.LightevalTask._original_get_docs(
            self_instance, limit, *args, **kwargs
        )
    else:
      # If no sample_range is configured, just fall back to the original logic.
      return lighteval_task.LightevalTask._original_get_docs(
          self_instance, max_samples, *args, **kwargs
      )

  # Replace the original class method with the slicing version.
  lighteval_task.LightevalTask.get_docs = slicing_get_docs


@registry.register_framework("lighteval")
class LightEvalFramework(base.AbstractEvalFramework):
  """Framework implementation leveraging lighteval for evaluations."""

  def evaluate(
      self,
      runner: "runners_base.AbstractRunner",
      tasks: list[str],
      limit: int | float | None = None,
      sample_range: tuple[int, int] | None = None,
      batch_size: int | None = None,
      eval_args: dict[str, Any] | None = None,
  ) -> base.EvalResults:
    """Evaluates a custom runner using Lighteval.

    This method uses `ChatScoreLightevalModel` for generation and scoring,
    targeting models accessible via a server endpoint.

    Args:
        runner: The runner instance providing the model name and server URL.
        tasks: A list of task names to evaluate.
        limit: Maximum samples per task.
        sample_range: Range of samples to evaluate.
        batch_size: Evaluation batch size.
        eval_args: Additional evaluation arguments for the Lighteval pipeline.

    Returns:
        The EvalResults containing the evaluation results.
    """
    # Resolve unified evaluation arguments into Lighteval-specific overrides.
    eval_params = self._from_unified_eval_args(
        limit, sample_range, batch_size, eval_args, limit_key="max_samples"
    )

    # Configure LiteLLM connector targeting the runner's server endpoint.
    config = litellm_model.LiteLLMModelConfig(
        model_name=f"openai/{runner.model_name}",
        api_key="not-needed",
        base_url=f"{runner.server_url}/v1",
        # Use a single concurrent request to avoid concurrency issues in the
        # LiteRT LM server.
        concurrent_requests=1,
    )

    # Use our custom model adapter that delegates generation and scoring.
    model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)
    return self._run_pipeline(
        tasks=tasks,
        model=model,
        limit=eval_params.limit,
        sample_range=eval_params.sample_range,
        eval_args=eval_params.eval_args,
    )

  def evaluate_native(
      self,
      model_config: base.NativeModelConfig,
      tasks: list[str],
      limit: int | float | None = None,
      sample_range: tuple[int, int] | None = None,
      batch_size: int | None = None,
      eval_args: dict[str, Any] | None = None,
  ) -> base.EvalResults:
    """Evaluates a model using Lighteval's native backends.

    This method supports models running on native engines recognized by
    Lighteval (e.g., Accelerate, vLLM, SGLang).

    Args:
        model_config: Configuration for the native model, including model type
          and arguments.
        tasks: A list of task names to evaluate.
        limit: Maximum samples per task.
        sample_range: Range of samples to evaluate, space-separated (e.g., (10,
          20)).
        batch_size: Evaluation batch size.
        eval_args: Additional evaluation arguments for the Lighteval pipeline.

    Returns:
        The EvalResults containing the evaluation results.
    """
    # Resolve unified evaluation arguments into Lighteval-specific overrides.
    eval_params = self._from_unified_eval_args(
        limit, sample_range, batch_size, eval_args, limit_key="max_samples"
    )

    # Retrieve the metadata required to instantiate the requested native engine.
    if model_config.model not in _NATIVE_RUNNER_CONFIGS:
      raise ValueError(f"Unsupported native model type: {model_config.model}")
    info = _NATIVE_RUNNER_CONFIGS[model_config.model]

    config_cls = getattr(
        importlib.import_module(info.module_path), info.config_class
    )

    runner_args = self._from_unified_runner_args(
        model_config,
        model_path_key=info.model_path_field_name or "model_path",
        device_key=info.device_field_name or "device",
    )

    # Avoid injecting invalid parameters if the native config doesn't support
    # them.
    if not info.model_path_field_name:
      runner_args.model_args.pop("model_path", None)
    if not info.device_field_name:
      runner_args.model_args.pop("device", None)

    # Batch size needs special handling because lighteval treats it as a model
    # argument.
    if info.batch_size_field_name:
      final_batch_size = self._consume_arg(
          eval_params.batch_size,
          runner_args.model_args,
          info.batch_size_field_name,
          "--batch-size",
          "--runner-args",
      )
      if final_batch_size is not None:
        runner_args.model_args[info.batch_size_field_name] = final_batch_size

    # Instantiate the engine-specific Lighteval configuration.
    lm_config = config_cls(**runner_args.model_args)
    return self._run_pipeline(
        tasks=tasks,
        model_config=lm_config,
        limit=eval_params.limit,
        sample_range=eval_params.sample_range,
        eval_args=eval_params.eval_args,
    )

  def _run_pipeline(
      self,
      tasks: list[str],
      model: Any = None,
      model_config: Any = None,
      limit: int | float | None = None,
      sample_range: tuple[int, int] | None = None,
      eval_args: dict[str, Any] | None = None,
  ) -> base.EvalResults:
    """Runs the Lighteval pipeline with the specified parameters.

    This internal helper abstracts the instantiation and execution of the
    lighteval `Pipeline`, bridging the gap between Custom Runners (which supply
    an instantiated `model`) and Native runners (which supply a `model_config`).

    Only one of `model` or `model_config` should be provided at a time:
      - Provide `model` when delegating execution to a pre-instantiated model
        adapter (e.g., a custom `LiteLLMClient` that talks to a runner server).
      - Provide `model_config` when triggering Lighteval's native engines
        (e.g., Accelerate, vLLM). Lighteval will use this config object to
        instantiate internal engine backends for evaluation itself.

    Args:
        tasks: A list of task names to evaluate.
        model: A pre-instantiated model adapter instance (used for custom
          evaluations). Defaults to None.
        model_config: A native Lighteval model configuration object (used for
          native pipeline execution). Defaults to None.
        limit: Maximum number of samples to evaluate per task if integer, or
          fraction of samples if float. Defaults to None.
        sample_range: Range of samples to evaluate.
        eval_args: Additional lightweight arguments for PipelineParameters.

    Returns:
        The EvalResults from the pipeline execution.
    """
    eval_args = dict(eval_args or {})

    configure_lighteval_slicing(limit, sample_range)
    if sample_range:
      max_samples = sample_range[1] + 1
    elif limit and isinstance(limit, int):
      max_samples = limit
    else:
      max_samples = None

    pipeline_params = lighteval_pipeline.PipelineParameters(
        launcher_type=lighteval_pipeline.ParallelismManager.CUSTOM,
        max_samples=max_samples,
        **eval_args,
    )
    pipeline = lighteval_pipeline.Pipeline(
        tasks=",".join(tasks),
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker.EvaluationTracker(
            output_dir="/tmp/lighteval"
        ),
        # Lighteval expects either `model` (an instantiated BaseModel subclass
        # like our custom LiteLLMClient) or `model_config` (for native engines).
        model=model,
        model_config=model_config,
    )
    # Suppress debug info and verbose logging to avoid polluting the output.
    old_suppress = litellm.suppress_debug_info
    old_verbose = litellm.set_verbose
    try:
      litellm.suppress_debug_info = True
      litellm.set_verbose = False
      pipeline.evaluate()
      pipeline.save_and_push_results()
    finally:
      litellm.suppress_debug_info = old_suppress
      litellm.set_verbose = old_verbose

    configure_lighteval_slicing(limit=None, sample_range=None)

    raw_results = pipeline.get_results()
    # Explicitly fetch the per-sample details as dictionaries.
    raw_results["samples"] = pipeline.evaluation_tracker.details

    return _to_eval_results(raw_results)

  @classmethod
  def supported_task_ids(cls) -> list[str]:
    """Returns a list of all task IDs supported by Lighteval.

    Note: This method accesses private members (`_task_registry` and
    `_task_superset_dict`) of the `lighteval.tasks.registry.Registry`
    class because Lighteval does not provide a public API to retrieve the full
    list of supported tasks and suite aliases.
    """
    registry_obj = _get_lighteval_registry()
    if not registry_obj:
      return []
    try:
      # Accessing private members is necessary here as no public API exists.
      concrete = getattr(registry_obj, "_task_registry", {})
      suites = getattr(registry_obj, "_task_superset_dict", {})
      return sorted(set(concrete) | set(suites))
    except AttributeError:
      return []

  @classmethod
  def subtasks_of(cls, task: str) -> list[str]:
    """Expands a lighteval suite (e.g. `mmlu`) to its concrete subtasks.

    Subtasks live in `Registry._task_superset_dict`, which already maps
    each suite name to its full `suite:subtask` list (suite layout is
    flat — no nested suites — so a single lookup is enough).

    Args:
        task: The parent task identifier to expand.

    Returns:
        A list of concrete subtask identifiers. Returns an empty list for
        unknown names or concrete tasks.
    """
    return sorted(_lighteval_suite_dict().get(task, []))

  @classmethod
  def supported_runners(cls) -> list[str]:
    """Returns a list of custom runners and native runners supported by the framework."""
    from model_eval.runners import registry as runner_registry  # pylint: disable=g-import-not-at-top

    custom_runners = runner_registry.get_all_runners()
    native_runners = list(_NATIVE_RUNNER_CONFIGS.keys())
    return sorted(list(set(custom_runners + native_runners)))

  @classmethod
  def describe_eval_args(cls) -> list[dict[str, Any]]:
    return introspection.get_fields(lighteval_pipeline.PipelineParameters)

  @classmethod
  def describe_native_runner_args(cls, runner: str) -> list[dict[str, Any]]:
    if runner in _NATIVE_RUNNER_CONFIGS:
      info = _NATIVE_RUNNER_CONFIGS[runner]

      return introspection.get_fields(
          getattr(importlib.import_module(info.module_path), info.config_class)
      )
    raise NotImplementedError


@functools.lru_cache(maxsize=1)
def _get_lighteval_registry() -> lighteval_registry.Registry | None:
  """Instantiates and caches the Lighteval Registry.

  Construction is cached because it is slow (loads hundreds of task configs
  from disk) and emits noisy stderr warnings.

  Returns:
      The initialized Registry instance, or None if Lighteval is not installed
      or initialization fails.
  """
  try:
    return lighteval_registry.Registry()
  except (ImportError, AttributeError):
    return None


@functools.lru_cache(maxsize=1)
def _lighteval_suite_dict() -> dict[str, list[str]]:
  """Returns lighteval's suite -> [colon-prefixed subtasks] mapping.

  Cached because constructing a `Registry` loads ~650 task configs from
  disk and emits a noisy stderr warning each time. The mapping is static
  for the process lifetime, so a single lookup is safe.

  Returns:
      A dictionary mapping suite names to lists of subtasks.
  """
  registry_obj = _get_lighteval_registry()
  if not registry_obj:
    return {}
  return getattr(registry_obj, "_task_superset_dict", {}) or {}


def _to_eval_results(raw: Any) -> base.EvalResults:
  return base.EvalResults(
      framework_type="lighteval",
      aggregated_metrics=raw.get("results", {}),
      per_sample_outputs=raw.get("samples", {}),
      metadata={"config": raw.get("config")},
  )
