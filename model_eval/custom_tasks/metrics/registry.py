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

"""Metric registry and compose function."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Union, overload
from model_eval.custom_tasks import base
from model_eval.custom_tasks.metrics import normalize

DatasetRow = base.DatasetRow
Normalizer = normalize.Normalizer


class MetricFn(Protocol):
  """Protocol for registered metric functions."""

  def __call__(
      self,
      preds: List[Any],
      gts: List[Any],
      rows: List[DatasetRow],
      *,
      normalizer: Normalizer,
  ) -> Dict[str, Any]:
    """Executes the metric evaluation on model predictions and ground truths.

    Args:
      preds: A list of candidate model prediction items.
      gts: A list of expected ground truth / reference items.
      rows: A list of input DatasetRow structs providing complete metadata
        context.
      normalizer: A string normalizer callable to sanitize text items before
        comparison.

    Returns:
      A dictionary of computed metric names and their corresponding numeric
      values.
    """
    ...


TaskMetricFn = Callable[
    [Iterable[Any], Iterable[Any], Iterable[DatasetRow]], Dict[str, Any]
]

_METRICS: Dict[str, MetricFn] = {}


@overload
def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
  ...


@overload
def register_metric(name: str, fn: MetricFn) -> MetricFn:
  ...


def register_metric(name: str, fn: Optional[MetricFn] = None) -> Any:
  """Registers a metric function under a unique name.

  Args:
    name: The string identifier for the metric.
    fn: Optional metric function to register. If provided, registers and returns
      it. If None, returns a decorator.

  Returns:
    The registered function, or a decorator that registers the function.

  Raises:
    ValueError: If the metric name is already registered.
  """

  def _reg(f: MetricFn) -> MetricFn:
    if name in _METRICS:
      raise ValueError(f"metric '{name}' already registered")
    _METRICS[name] = f
    return f

  if fn is not None:
    return _reg(fn)
  return _reg


def get_metric(name: str) -> MetricFn:
  """Retrieves a registered metric function by its name.

  Args:
    name: The name of the registered metric.

  Returns:
    The registered MetricFn.

  Raises:
    KeyError: If the metric name is not found in the registry.
  """
  if name not in _METRICS:
    raise KeyError(f"unknown metric '{name}'; have {sorted(_METRICS)}")
  return _METRICS[name]


def list_metrics() -> List[str]:
  """Returns a sorted list of all registered metric names.

  Returns:
    A list of registered metric names.
  """
  return sorted(_METRICS)


def clear_metrics() -> None:
  """Clears all registered metrics, restoring the registry to its initial empty state."""
  _METRICS.clear()


def compose(
    metric_names: List[str], normalizer: Optional[str] = None
) -> TaskMetricFn:
  """Builds a composite metric function from multiple registered metric names.

  The resulting function runs all specified metrics and merges their output
  dictionaries.

  Args:
    metric_names: A list of registered metric names to execute.
    normalizer: An optional registered normalizer name to pass to the metrics.

  Returns:
    A TaskMetricFn compatible with CustomTask.metric_fn.

  Raises:
    ValueError: If metric_names is empty.
    KeyError: If any metric or normalizer name is unknown.
  """
  if not metric_names:
    raise ValueError("compose() needs >=1 metric")
  norm = normalize.get_normalizer(normalizer)
  metrics_list = [(n, get_metric(n)) for n in metric_names]

  def metric_fn(
      preds: Iterable[Any], gts: Iterable[Any], rows: Iterable[DatasetRow]
  ) -> Dict[str, Any]:
    preds_list = list(preds)
    gts_list = list(gts)
    rows_list = list(rows)
    if not (len(preds_list) == len(gts_list) == len(rows_list)):
      raise ValueError(
          "Mismatched input lengths for evaluation:"
          f" predictions ({len(preds_list)}), ground truths ({len(gts_list)}),"
          f" rows ({len(rows_list)})."
      )
    out = {}
    for name, m in metrics_list:
      res = m(preds_list, gts_list, rows_list, normalizer=norm)
      overlap = set(out.keys()) & set(res.keys())
      if overlap:
        raise ValueError(
            f"Key collision detected during metric composition: {overlap} "
            f"from metric '{name}' already exists."
        )
      out.update(res)
    return out

  return metric_fn
