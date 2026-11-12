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

"""Custom evaluation framework implementation."""

from collections.abc import Iterable, Iterator
import contextlib
import itertools
import json
import math
import os
import sys
import time
from typing import Any, TypeVar, cast

from model_eval.custom_tasks import base as tasks_base
from model_eval.custom_tasks import loaders
from model_eval.custom_tasks import registry as tasks_registry
from model_eval.frameworks import base
from model_eval.frameworks import registry
from model_eval.runners import base as runners_base
from model_eval.utils import introspection
import httpx
import tqdm

# Per-request timeout used unless the runner sets `request_timeout_sec`.
_DEFAULT_REQUEST_TIMEOUT_SEC = 120.0
# Error recorded for requests after an earlier request of the same row failed.
_SKIPPED_ERROR = "skipped: an earlier request failed"
# Maximum number of response body characters kept in a recorded error.
_MAX_ERROR_CHARS = 500

_T = TypeVar("_T")

# Framework-level eval args consumed here (not forwarded as task args).
INFERENCE_ONLY_ARG = "inference_only"
SCORE_FROM_SAMPLES_ARG = "score_from_samples"


def _apply_limit(rows: Iterable[_T], limit: int | float) -> Iterator[_T]:
  """Slices a dataset using an absolute count limit or a percentage fraction.

  An absolute limit is applied lazily, so rows past the limit are never
  produced. A fractional limit needs the dataset size and loads all rows.

  Args:
      rows: The input dataset rows to sample from.
      limit: Absolute integer count or a float representing a fraction.

  Returns:
      An iterator over the sliced rows.
  """
  if isinstance(limit, float) and limit < 1:
    rows_list = list(rows)
    return iter(rows_list[: int(math.ceil(len(rows_list) * limit))])
  return itertools.islice(rows, int(limit))


def _apply_samples(
    rows: Iterable[_T], samples_val: Any, task_name: str
) -> Iterator[_T]:
  """Filters a dataset by keeping precise row indices or resolving range slices.

  Index lists, ranges ("start-end") and comma-separated indices are applied
  lazily, stopping after the largest requested index. Slice expressions
  ("start:stop:step") may be open-ended and load all rows.

  Args:
      rows: The dataset rows to slice from.
      samples_val: A dictionary mapping task names to indices/ranges, a list of
        indices, or a slice expression string.
      task_name: The specific name of the current task being evaluated.

  Raises:
      ValueError: If samples_val is not a dict, str, or list.

  Returns:
      An iterator over the filtered dataset rows, in dataset order.
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
      return iter(rows)
  else:
    task_expr = samples_val

  if isinstance(task_expr, list):
    if any(i < 0 for i in task_expr):
      # Negative indices count from the end, which needs all rows.
      rows_list = list(rows)
      return (
          rows_list[i]
          for i in task_expr
          if -len(rows_list) <= i < len(rows_list)
      )
    indices = task_expr
  elif ":" in str(task_expr):
    rows_list = list(rows)
    return (
        rows_list[i]
        for i in loaders.parse_samples(str(task_expr), len(rows_list))
    )
  else:
    indices = loaders.parse_samples(str(task_expr), sys.maxsize)
  if not indices:
    return iter(())
  head = itertools.islice(rows, max(indices) + 1)
  if indices != sorted(set(indices)):
    # Unsorted or repeated indices keep the requested order, so the rows up
    # to the largest index are buffered.
    head_list = list(head)
    return (head_list[i] for i in indices if i < len(head_list))
  wanted = set(indices)
  return (row for i, row in enumerate(head) if i in wanted)


def _select_rows(
    rows: Iterable[_T],
    task_name: str,
    limit: int | float | None,
    sample_range: tuple[int, int] | None,
    samples: Any,
) -> Iterator[_T]:
  """Validates slicing options and returns an iterator over selected rows."""
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
  if limit is not None:
    return _apply_limit(rows, limit)  # pyrefly: ignore[bad-argument-type, bad-return]
  if samples is not None:
    return _apply_samples(rows, samples, task_name)  # pyrefly: ignore[bad-argument-type, bad-return]
  return iter(rows)


def _post_all(
    http_client: httpx.Client, url: str, requests: tasks_base.Requests
) -> tuple[
    list[Any], list[str | None], list[tasks_base.RequestTelemetry | None]
]:
  """POSTs each JSON request body in `requests` to `url` in order.

  Requests in a row often depend on each other (e.g. index, then query), so
  once one fails the remaining requests in that row are skipped. Failures are
  recorded rather than raised so a single bad row does not abort a long run.

  Args:
    http_client: HTTP client used to issue POST requests.
    url: Full endpoint URL (`{server_url}/{endpoint}`).
    requests: List of JSON request bodies for the row.

  Returns:
    A tuple `(responses, errors, telemetry)` of equal length:
    - `responses[i]` is the decoded JSON response on success, or None on
      failure/skip.
    - `errors[i]` is None on success, or a description of the failure/skip.
    - `telemetry[i]` is a `RequestTelemetry` dict (`{"latency_sec": ...}` with
      the client-observed wall-clock seconds spent on the request, including
      transport), or None if it was skipped.
  """
  responses: list[Any] = []
  errors: list[str | None] = []
  telemetry: list[tasks_base.RequestTelemetry | None] = []
  failed = False
  for body in requests:
    if failed:
      responses.append(None)
      errors.append(_SKIPPED_ERROR)
      telemetry.append(None)
      continue
    error: str | None = None
    start = time.perf_counter()
    try:
      resp = http_client.post(url, json=body)
      if resp.is_error:
        error = f"HTTP {resp.status_code}: {resp.text[:_MAX_ERROR_CHARS]}"
      else:
        responses.append(resp.json())
        errors.append(None)
    except (httpx.HTTPError, ValueError) as e:
      # ValueError covers a response body that is not valid JSON.
      error = f"{type(e).__name__}: {e}"
    telemetry.append({"latency_sec": time.perf_counter() - start})
    if error is not None:
      failed = True
      responses.append(None)
      errors.append(error)
  return responses, errors, telemetry


def _with_telemetry(
    row: tasks_base.DatasetRow,
    telemetry: list[tasks_base.RequestTelemetry | None],
) -> tasks_base.DatasetRow:
  """Returns a shallow copy of `row` with `metadata["telemetry"]` set.

  Copying keeps the dataset's own row objects untouched while exposing the
  per-request telemetry to `metric_fn` and `sample_input_fn`.

  Args:
    row: The dataset row that was just evaluated.
    telemetry: Per-request telemetry dicts as returned by `_post_all`.

  Returns:
    A shallow copy of `row` with `metadata["telemetry"]` set to `telemetry`.
  """
  copied: dict[str, Any] = dict(row)
  copied["metadata"] = {
      **(row.get("metadata") or {}),
      "telemetry": telemetry,
  }
  return cast(tasks_base.DatasetRow, copied)


def _as_bool(value: Any) -> bool:
  """Interprets eval-arg values such as `True`, `"true"` or `"1"`."""
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
      return True
    if normalized in ("0", "false", "no", "off", ""):
      return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean.")
  return bool(value)


def _open_text(path: str) -> Any:
  """Opens `path` for reading; remote (e.g. CNS) paths go through gfile."""
  if not os.path.exists(path):
    raise FileNotFoundError(f"Samples file does not exist: {path}")
  return open(path, "r", encoding="utf-8")


def _load_samples_file(path: str) -> dict[str, list[dict[str, Any]]]:
  """Loads a samples JSONL written by the pipeline, grouped by task name.

  Args:
    path: Path to a `run_<ts>_samples.jsonl` file.

  Raises:
    FileNotFoundError: If `path` does not exist.

  Returns:
    Mapping from task name to its sample records, in file order.
  """
  by_task: dict[str, list[dict[str, Any]]] = {}
  with _open_text(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      by_task.setdefault(str(rec.get("task", "")), []).append(rec)
  return by_task


_EvaluatedRow = tuple[
    tasks_base.DatasetRow,
    list[Any],
    list[str | None],
    Any,
    dict[str, Any] | None,
    list[tasks_base.RequestTelemetry | None] | None,
]


def _run_metric_fn(
    task: tasks_base.CustomTask,
    evaluated: list[_EvaluatedRow],
    task_args: tasks_base.TaskArgs,
) -> dict[str, float]:
  """Applies `task.metric_fn` to evaluated rows and `task_args`."""
  return task.metric_fn(
      (pred for _, pred, _, _, _, _ in evaluated),
      (row["ground_truth"] for row, _, _, _, _, _ in evaluated),
      (row for row, _, _, _, _, _ in evaluated),
      task_args,
  )


def _sample_outputs(
    evaluated: list[_EvaluatedRow],
) -> list[dict[str, Any]]:
  """Builds per-sample output records (the samples JSONL schema)."""
  outputs = []
  for row, pred, errors, sample_input, raw_metadata, telemetry in evaluated:
    out: dict[str, Any] = {
        "input": sample_input,
        "prediction": pred,
        "errors": errors,
    }
    if telemetry is not None:
      out["telemetry"] = telemetry
    out["ground_truth"] = row["ground_truth"]
    if raw_metadata is not None:
      out["metadata"] = raw_metadata
    outputs.append(out)
  return outputs


@registry.register_framework("custom")
class CustomFramework(base.AbstractEvalFramework):
  """Framework driving custom tasks against a runner's HTTP endpoint."""

  def evaluate(
      self,
      runner: runners_base.AbstractRunner | None,
      tasks: list[str],
      limit: int | float | None = None,
      sample_range: tuple[int, int] | None = None,
      batch_size: int | None = None,
      eval_args: dict[str, Any] | None = None,
  ) -> base.EvalResults:
    """Evaluates the runner across the specified custom tasks.

    Each row's `requests` are POSTed as JSON bodies in order to the task's
    `endpoint` (joined to `runner.server_url`).

    Two staged modes are supported through framework-level eval args:

    * `inference_only=True`: requests are sent and per-sample outputs are
      collected, but `metric_fn` is not run. The samples file written by the
      pipeline then serves as a cache of model outputs.
    * `score_from_samples=<path>`: no runner is used; rows, predictions and
      errors are reloaded from a previously written samples JSONL and only
      `metric_fn` runs. Remaining eval args override the cached `task_args` in
      each row's metadata, so scoring can be re-run with different settings.

    Args:
      runner: The target runner implementation responsible for server inference,
        or None when `score_from_samples` is set.
      tasks: A list of unique task names to look up and evaluate.
      limit: Optional limit on number of samples per task to generate.
      sample_range: Optional range of samples to evaluate.
      batch_size: Evaluation batch size, used for conflict checks/parity.
      eval_args: Optional evaluation arguments. `samples` is consumed by the
        framework; all remaining entries are forwarded as `task_args` to each
        task's `dataset` and `metric_fn`, merged on top of `runner.server_args`
        (eval args take precedence on key collisions).

    Raises:
      ValueError: If conflicting limit/sample options are provided.

    Returns:
      An EvalResults instance containing all aggregated task metrics and
      outputs. The per-sample outputs contain the keys 'input', 'prediction',
      'errors', 'telemetry' (when run against a server or present in the
      samples file), 'ground_truth', and optional 'metadata' (plus
      'sample_index' when preserved from a re-scored samples file); 'input' is
      `task.sample_input_fn(row)` when set, else the row's `requests`.
    """
    eval_params = self._from_unified_eval_args(
        limit, sample_range, batch_size, eval_args
    )
    inference_only = _as_bool(
        eval_params.eval_args.pop(INFERENCE_ONLY_ARG, False)
    )
    score_from_samples = eval_params.eval_args.pop(SCORE_FROM_SAMPLES_ARG, None)
    if inference_only and score_from_samples:
      raise ValueError(
          f"Cannot set both '{INFERENCE_ONLY_ARG}' and"
          f" '{SCORE_FROM_SAMPLES_ARG}'."
      )
    samples = eval_params.eval_args.pop("samples", None)
    reg = tasks_registry.TaskRegistry.global_registry()
    all_results = {}
    all_samples = {}

    if score_from_samples:
      # Scoring-only stage: no runner or HTTP traffic is involved.
      records_by_task = _load_samples_file(str(score_from_samples))
      allow_single_task_fallback = len(tasks) == 1
      for name in tasks:
        task = reg.get_task(name)
        all_results[name], all_samples[name] = self._score_samples(
            task,
            records_by_task,
            limit=eval_params.limit,
            sample_range=eval_params.sample_range,
            samples=samples,
            task_args=eval_params.eval_args,
            allow_single_task_fallback=allow_single_task_fallback,
        )
      return base.EvalResults(
          framework_type="custom",
          aggregated_metrics=all_results,
          per_sample_outputs=all_samples,
          metadata={SCORE_FROM_SAMPLES_ARG: str(score_from_samples)},
      )

    if runner is None:
      raise ValueError(
          f"A runner is required unless '{SCORE_FROM_SAMPLES_ARG}' is provided."
      )
    task_args = eval_params.eval_args
    timeout_sec = runner.request_timeout_sec or _DEFAULT_REQUEST_TIMEOUT_SEC
    with httpx.Client(timeout=timeout_sec) as http_client:
      for name in tasks:
        task = reg.get_task(name)
        all_results[name], all_samples[name] = self._run_task(
            runner,
            task,
            http_client,
            limit=eval_params.limit,
            sample_range=eval_params.sample_range,
            samples=samples,
            task_args=task_args,
            inference_only=inference_only,
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
      task_args: tasks_base.TaskArgs | None = None,
      inference_only: bool = False,
  ) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Executes a single custom evaluation task against `task.endpoint`.

    Args:
      runner: The target runner exposing `server_url` and `server_args`.
      task: The CustomTask instance defining the endpoint, dataset, and metrics.
      http_client: Explicit httpx client instance to use for API calls.
      limit: Optional limit on number of samples per task to generate.
      sample_range: Optional range of samples to evaluate.
      samples: Optional explicit sample indices or dictionary map.
      task_args: Task-specific arguments, merged on top of `runner.server_args`
        and forwarded to `task.dataset` and `task.metric_fn`.
      inference_only: If True, skip `task.metric_fn` and return empty metrics.

    Raises:
      ValueError: If conflicting limit/sample options are provided.

    Returns:
      A tuple `(metrics, sample_outputs)` where each entry in `sample_outputs`
      contains `'input'`, `'prediction'`, `'errors'`, `'telemetry'`,
      `'ground_truth'`, and optional `'metadata'`.
    """
    url = f"{runner.server_url}/{task.endpoint}"
    evaluated: list[_EvaluatedRow] = []

    # Callable datasets and `metric_fn` receive the runner's server args (e.g.
    # `model_name`) beneath the task args from --eval-args; eval args win on
    # collision. File datasets cannot consume args, so only eval args are
    # passed to the loader (and rejected if non-empty).
    merged_task_args: tasks_base.TaskArgs = {
        **runner.server_args,
        **(task_args or {}),
    }
    dataset_args = (
        merged_task_args if callable(task.dataset) else dict(task_args or {})
    )

    with contextlib.closing(
        loaders.load_dataset(task.dataset, dataset_args)
    ) as dataset:
      rows = _select_rows(dataset, task.name, limit, sample_range, samples)
      for row in tqdm.tqdm(rows, desc=f"Evaluating {task.name}"):
        raw_metadata = row.get("metadata")
        responses, errors, telemetry = _post_all(
            http_client, url, row["requests"]
        )
        row = _with_telemetry(row, telemetry)
        sample_input = (
            task.sample_input_fn(row)
            if task.sample_input_fn is not None
            else row["requests"]
        )
        evaluated.append(
            (row, responses, errors, sample_input, raw_metadata, telemetry)
        )

    if inference_only:
      result: dict[str, float] = {}
    else:
      result = _run_metric_fn(task, evaluated, merged_task_args)
    return result, _sample_outputs(evaluated)

  def _score_samples(
      self,
      task: tasks_base.CustomTask,
      records_by_task: dict[str, list[dict[str, Any]]],
      *,
      limit: int | float | None = None,
      sample_range: tuple[int, int] | None = None,
      samples: Any = None,
      task_args: tasks_base.TaskArgs | None = None,
      allow_single_task_fallback: bool = True,
  ) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Runs `task.metric_fn` over records reloaded from a samples file.

    Args:
      task: The CustomTask whose metric function is applied.
      records_by_task: Sample records grouped by task name, as produced by
        `_load_samples_file`.
      limit: Optional limit on number of samples to score.
      sample_range: Optional range of samples to score.
      samples: Optional explicit sample indices or dictionary map.
      task_args: Eval args that override the cached `task_args` in each row's
        metadata (e.g. to re-score with a different metric configuration).
      allow_single_task_fallback: Whether a single-task samples file may be
        scored under a different task name when only one task is requested.

    Raises:
      ValueError: If the samples file has no records for the task.

    Returns:
      A tuple `(metrics, sample_outputs)` with the same shape as `_run_task`.
    """
    records = records_by_task.get(task.name)
    if (
        records is None
        and allow_single_task_fallback
        and len(records_by_task) == 1
    ):
      # A single-task samples file may be scored under a different task name.
      records = next(iter(records_by_task.values()))
    if not records:
      raise ValueError(
          f"No samples for task '{task.name}' in samples file; found tasks:"
          f" {sorted(records_by_task)}"
      )
    selected_records = list(
        _select_rows(records, task.name, limit, sample_range, samples)
    )
    overrides = dict(task_args or {})
    cached_task_args: tasks_base.TaskArgs = {}
    if selected_records:
      first_meta = selected_records[0].get("metadata") or {}
      cached_task_args = dict(first_meta.get("task_args") or {})
    effective_task_args: tasks_base.TaskArgs = {
        **cached_task_args,
        **overrides,
    }
    evaluated: list[_EvaluatedRow] = []
    for rec in selected_records:
      raw_metadata = rec.get("metadata")
      telemetry = rec.get("telemetry")
      metadata = dict(raw_metadata) if raw_metadata is not None else {}
      if overrides:
        metadata["task_args"] = {
            **(metadata.get("task_args") or {}),
            **overrides,
        }
      saved_input = rec.get("input")
      requests: tasks_base.Requests = (
          saved_input if isinstance(saved_input, list) else []
      )
      row_dict: dict[str, Any] = {
          "requests": requests,
          "ground_truth": rec.get("ground_truth"),
      }
      if raw_metadata is not None or metadata:
        row_dict["metadata"] = metadata
      row = cast(tasks_base.DatasetRow, row_dict)
      if telemetry is not None:
        row = _with_telemetry(row, telemetry)
      prediction: list[Any] = list(rec.get("prediction") or [])
      errors: list[str | None] = list(rec.get("errors") or [])
      sample_input = saved_input if saved_input is not None else []
      user_meta_out = (
          dict(metadata)
          if (raw_metadata is not None or overrides)
          else None
      )
      evaluated.append(
          (row, prediction, errors, sample_input, user_meta_out, telemetry)
      )
    outputs = _sample_outputs(evaluated)
    for rec, out in zip(selected_records, outputs):
      if rec.get("sample_index") is not None:
        out["sample_index"] = rec["sample_index"]
    return _run_metric_fn(task, evaluated, effective_task_args), outputs

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
