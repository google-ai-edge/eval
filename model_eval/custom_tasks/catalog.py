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

"""Catalog loading and declarative TaskSpec definition layer."""

import itertools
import pathlib
from typing import Any, Dict, List, Optional
from model_eval.config.generation_config import GenerationConfig
from model_eval.custom_tasks import base
from model_eval.custom_tasks import loaders
from model_eval.custom_tasks import metrics
from model_eval.custom_tasks import registry
import pydantic
import yaml


class TaskSpec(pydantic.BaseModel):
  """Declarative specification for a custom evaluation task.

  Attributes:
    name: Unique task identifier.
    loader: Registered dataset loader name.
    metrics: List of registered metric names.
    generation_config: Generation parameters (e.g., temperature,
      max_new_tokens). This field is required; use an empty block (e.g., `{}`)
      to fall back to defaults.
    normalizer: Optional registered normalizer name.
    options: Configuration dictionary passed to the loader.
  """

  model_config = pydantic.ConfigDict(extra="allow")

  name: str
  loader: str
  metrics: List[str]
  generation_config: GenerationConfig
  normalizer: Optional[str] = None
  options: Dict[str, Any] = {}

  @pydantic.model_validator(mode="after")
  def _validate(self) -> "TaskSpec":
    if self.loader not in loaders.list_loaders():
      raise ValueError(f"{self.name}: unknown loader '{self.loader}'")
    bad = [m for m in self.metrics if m not in metrics.list_metrics()]
    if bad:
      raise ValueError(f"{self.name}: unknown metric(s) {bad}")
    return self


def _render(obj: Any, ctx: Dict[str, Any]) -> Any:
  """Recursively renders format strings within an object using a context dictionary."""
  if isinstance(obj, str):
    return obj.format(**ctx)
  if isinstance(obj, dict):
    return {k: _render(v, ctx) for k, v in obj.items()}
  if isinstance(obj, list):
    return [_render(v, ctx) for v in obj]
  return obj


def expand_for_each(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
  """Expands a raw dictionary containing a 'for_each' key into multiple Cartesian permutations."""
  raw = dict(raw)
  fe = raw.pop("for_each", None)
  if not fe:
    return [raw]
  keys = list(fe)
  return [
      _render(raw, dict(zip(keys, combo)))
      for combo in itertools.product(*[fe[k] for k in keys])
  ]


def load_catalog(path: str, spec_cls: Any = TaskSpec) -> List[Any]:
  """Loads, expands, and validates task specifications from a YAML file."""
  raw_list = yaml.safe_load(pathlib.Path(path).read_text()) or []
  specs = []
  for raw in raw_list:
    for expanded in expand_for_each(raw):
      specs.append(spec_cls(**expanded))
  names = [s.name for s in specs]
  dupes = {n for n in names if names.count(n) > 1}
  if dupes:
    raise ValueError(f"duplicate task names in catalog: {sorted(dupes)}")
  return specs


def build_task(spec: Any) -> base.CustomTask:
  """Constructs a concrete CustomTask from a TaskSpec."""
  loader = loaders.get_loader(spec.loader)
  return base.CustomTask(
      name=spec.name,
      dataset=(lambda s=spec, l=loader: l(s)),
      metric_fn=metrics.compose(spec.metrics, spec.normalizer),
      generation_config=spec.generation_config,
  )


def register_specs(
    specs: List[Any], registry_instance: Optional[Any] = None
) -> None:
  """Registers concrete CustomTask instances built from specifications."""
  reg = registry_instance or registry.TaskRegistry.global_registry()
  for spec in specs:
    reg.register(build_task(spec))


def register_catalog(
    path: str,
    registry_instance: Optional[Any] = None,
    spec_cls: Any = TaskSpec,
) -> None:
  """Loads a YAML catalog and registers all its tasks into the task registry."""
  register_specs(load_catalog(path, spec_cls), registry_instance)
