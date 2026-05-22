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
import importlib
from typing import Any

from model_eval.frameworks import base
from model_eval.frameworks import registry
from model_eval.frameworks.lighteval import _chat_score_lighteval_model
from model_eval.runners import base as runners_base
from model_eval.utils import introspection
from lighteval import pipeline as lighteval_pipeline  # type: ignore
from lighteval.logging import evaluation_tracker  # type: ignore
from lighteval.models.endpoints import litellm_model  # type: ignore
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


@registry.register_framework("lighteval")
class LightEvalFramework(base.AbstractEvalFramework):
  """Framework implementation leveraging lighteval for evaluations."""

  def evaluate(
      self,
      runner: "runners_base.AbstractRunner",
      tasks: list[str],
      limit: int | float | None = None,
      batch_size: int | None = None,
      eval_args: dict[str, Any] | None = None,
  ) -> base.EvalResults:
    """Evaluates a custom runner using Lighteval.

    This method uses `ChatScoreLightevalModel` for generation and scoring,
    targeting models accessible via a server endpoint.

    Args:
        runner: The runner instance providing the model name and server URL.
        tasks: A list of task names to evaluate.
        limit: Maximum samples per task. Must be an integer or None.
        batch_size: Evaluation batch size.
        eval_args: Additional evaluation arguments for the Lighteval pipeline.

    Returns:
        The EvalResults containing the evaluation results.

    Raises:
        ValueError: If limit is a fraction (float < 1), as Lighteval only
          supports integer limits.
    """
    # Resolve unified evaluation arguments into Lighteval-specific overrides.
    eval_params = self._from_unified_eval_args(
        limit, batch_size, eval_args, limit_key="max_samples"
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
    max_samples = self._resolve_max_samples(eval_params.limit)
    # Use our custom model adapter that delegates generation and scoring.
    model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)
    return self._run_pipeline(
        tasks=tasks,
        model=model,
        max_samples=max_samples,
        eval_args=eval_params.eval_args,
    )

  def evaluate_native(
      self,
      model_config: base.NativeModelConfig,
      tasks: list[str],
      limit: int | float | None = None,
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
        limit: Maximum samples per task. Must be an integer or None.
        batch_size: Evaluation batch size.
        eval_args: Additional evaluation arguments for the Lighteval pipeline.

    Returns:
        The EvalResults containing the evaluation results.

    Raises:
        ValueError: If the native model type is unsupported, or if limit is a
          fraction (float < 1).
    """
    # Resolve unified evaluation arguments into Lighteval-specific overrides.
    eval_params = self._from_unified_eval_args(
        limit, batch_size, eval_args, limit_key="max_samples"
    )
    max_samples = self._resolve_max_samples(eval_params.limit)

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
        max_samples=max_samples,
        eval_args=eval_params.eval_args,
    )

  def _resolve_max_samples(self, limit: int | float | None) -> int | None:
    """Resolves the maximum number of samples to evaluate.

    Args:
        limit: The unified limit argument specifying maximum samples.

    Returns:
        The integer number of maximum samples to evaluate, or None if unbounded.

    Raises:
        ValueError: If the limit is provided as a fractional value.
    """
    if limit is not None and isinstance(limit, float) and limit < 1:
      raise ValueError(
          "Lighteval does not support fractional limits. Please provide an"
          " integer limit or use the 'max_samples' key in --eval-args."
      )
    return int(limit) if limit is not None else None

  def _run_pipeline(
      self,
      tasks: list[str],
      model: Any = None,
      model_config: Any = None,
      max_samples: int | float | None = None,
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
        max_samples: The maximum number of samples to evaluate, or None.
        eval_args: Additional lightweight arguments for PipelineParameters.

    Returns:
        The EvalResults from the pipeline execution.
    """
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

    raw_results = pipeline.get_results()
    # Explicitly fetch the per-sample details as dictionaries.
    raw_results["samples"] = pipeline.evaluation_tracker.details

    return _to_eval_results(raw_results)

  @classmethod
  def supported_task_ids(cls) -> list[str]:
    try:
      registry_obj = lighteval_registry.Registry()
      return list(getattr(registry_obj, "_task_registry", {}).keys())
    except ImportError:
      return []

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


def _to_eval_results(raw: Any) -> base.EvalResults:
  return base.EvalResults(
      framework_type="lighteval",
      aggregated_metrics=raw.get("results", {}),
      per_sample_outputs=raw.get("samples", {}),
      metadata={"config": raw.get("config")},
  )
