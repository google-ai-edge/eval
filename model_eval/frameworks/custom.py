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
import math
import sys
from typing import Any

from model_eval.custom_tasks import base as tasks_base
from model_eval.custom_tasks import loaders
from model_eval.custom_tasks import registry as tasks_registry
from model_eval.frameworks import base
from model_eval.frameworks import registry
from model_eval.frameworks import utils
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


def _apply_limit(
    rows: Iterable[dict[str, Any]], limit: int | float
) -> Iterator[dict[str, Any]]:
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
    rows: Iterable[dict[str, Any]], samples_val: Any, task_name: str
) -> Iterator[dict[str, Any]]:
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
      return (rows_list[i] for i in task_expr if i < len(rows_list))
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
    rows: Iterable[tasks_base.DatasetRow],
    task_name: str,
    limit: int | float | None,
    sample_range: tuple[int, int] | None,
    samples: Any,
) -> Iterator[tasks_base.DatasetRow]:
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
  if limit:
    return _apply_limit(rows, limit)  # pyrefly: ignore[bad-argument-type, bad-return]
  if samples:
    return _apply_samples(rows, samples, task_name)  # pyrefly: ignore[bad-argument-type, bad-return]
  return iter(rows)


def _post_all(
    http_client: httpx.Client, url: str, requests: tasks_base.Requests
) -> tuple[list[Any], list[str | None]]:
  """POSTs each JSON request body in `requests` to `url` in order.

  Requests in a row often depend on each other (e.g. index, then query), so
  once one fails the remaining requests in that row are skipped. Failures are
  recorded rather than raised so a single bad row does not abort a long run.

  Args:
    http_client: HTTP client used to issue POST requests.
    url: Full endpoint URL (`{server_url}/{endpoint}`).
    requests: List of JSON request bodies for the row.

  Returns:
    A tuple `(responses, errors)` of equal length:
    - `responses[i]` is the decoded JSON response on success, or None on
      failure/skip.
    - `errors[i]` is None on success, or a description of the failure/skip.
  """
  responses: list[Any] = []
  errors: list[str | None] = []
  failed = False
  for body in requests:
    if failed:
      responses.append(None)
      errors.append(_SKIPPED_ERROR)
      continue
    error: str | None = None
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
    if error is not None:
      failed = True
      responses.append(None)
      errors.append(error)
  return responses, errors


@registry.register_framework("custom")
class CustomFramework(base.AbstractEvalFramework):
  """Framework driving custom tasks against a runner's HTTP endpoint."""

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

    Each row's `requests` are POSTed as JSON bodies in order to the target
    `endpoint` specified in `eval_args` (which must be served by `runner`).

    Args:
      runner: The target runner implementation responsible for server inference.
      tasks: A list of unique task names to look up and evaluate.
      limit: Optional limit on number of samples per task to generate.
      sample_range: Optional range of samples to evaluate.
      batch_size: Evaluation batch size, used for conflict checks/parity.
      eval_args: Evaluation arguments. Must contain `endpoint` (the server path
        to POST each row's requests to). `samples` is also consumed by the
        framework; all remaining entries are forwarded as `task_args` to each
        task's `dataset`.

    Raises:
      ValueError: If `eval_args['endpoint']` is not provided or not in
        `runner.endpoints`, or if conflicting limit/sample options are provided.

    Returns:
      An EvalResults instance containing all aggregated task metrics and
      outputs. The per-sample outputs contain the keys 'input', 'prediction',
      'errors', and 'ground_truth'.
    """
    eval_params = self._from_unified_eval_args(
        limit, sample_range, batch_size, eval_args
    )
    endpoint = utils.validate_endpoint(
        runner, eval_params.eval_args.pop("endpoint", None), "custom"
    )
    samples = eval_params.eval_args.pop("samples", None)
    task_args = eval_params.eval_args

    reg = tasks_registry.TaskRegistry.global_registry()
    all_results = {}
    all_samples = {}
    timeout_sec = runner.request_timeout_sec or _DEFAULT_REQUEST_TIMEOUT_SEC
    with httpx.Client(timeout=timeout_sec) as http_client:
      for name in tasks:
        task = reg.get_task(name)
        all_results[name], all_samples[name] = self._run_task(
            runner,
            endpoint,
            task,
            http_client,
            limit=eval_params.limit,
            sample_range=eval_params.sample_range,
            samples=samples,
            task_args=task_args,
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
      endpoint: str,
      task: tasks_base.CustomTask,
      http_client: httpx.Client,
      *,
      limit: int | float | None = None,
      sample_range: tuple[int, int] | None = None,
      samples: Any = None,
      task_args: tasks_base.TaskArgs | None = None,
  ) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Executes a single custom evaluation task against `endpoint`.

    Args:
      runner: The target runner exposing `server_url`.
      endpoint: Server path (without slashes) to POST each request to.
      task: The CustomTask instance defining the dataset and metric hooks.
      http_client: Explicit httpx client instance to use for API calls.
      limit: Optional limit on number of samples per task to generate.
      sample_range: Optional range of samples to evaluate.
      samples: Optional explicit sample indices or dictionary map.
      task_args: Task-specific arguments, forwarded to `task.dataset`.

    Raises:
      ValueError: If conflicting limit/sample options are provided.

    Returns:
      A tuple `(metrics, sample_outputs)` where each entry in `sample_outputs`
      contains `'input'`, `'prediction'`, `'errors'`, and `'ground_truth'`.
    """
    url = f"{runner.server_url}/{endpoint}"
    evaluated: list[
        tuple[tasks_base.DatasetRow, list[Any], list[str | None]]
    ] = []

    with contextlib.closing(
        loaders.load_dataset(task.dataset, dict(task_args or {}))
    ) as dataset:
      rows = _select_rows(dataset, task.name, limit, sample_range, samples)
      for row in tqdm.tqdm(rows, desc=f"Evaluating {task.name}"):
        responses, errors = _post_all(http_client, url, row["requests"])
        evaluated.append((row, responses, errors))

    result = task.metric_fn(
        (pred for _, pred, _ in evaluated),
        (row["ground_truth"] for row, _, _ in evaluated),
        (row for row, _, _ in evaluated),
    )
    sample_outputs = [
        {
            "input": row["requests"],
            "prediction": pred,
            "errors": errors,
            "ground_truth": row["ground_truth"],
        }
        for row, pred, errors in evaluated
    ]
    return result, sample_outputs

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
