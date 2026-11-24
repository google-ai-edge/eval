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

"""Registry for runner implementations."""

import importlib
import typing
from typing import Any, ClassVar, Protocol, TypeVar

from model_eval.runners import base
import pydantic


class _HasConfig(Protocol):
  config_type: ClassVar[type[pydantic.BaseModel]]

  def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:  # pylint: disable=unused-argument
    ...


T = TypeVar("T", bound=_HasConfig)

_REGISTRY: dict[str, type[_HasConfig]] = {}

# Runners imported on first use, so their dependencies stay optional.
_LAZY_IMPORTS: dict[str, str] = {
}


def _import_lazy_runner(name: base.RunnerType | str) -> None:
  """Imports the module that registers runner `name`, if it is lazy."""
  if name not in _REGISTRY and str(name) in _LAZY_IMPORTS:
    importlib.import_module(_LAZY_IMPORTS[str(name)])


def register_runner(name: base.RunnerType | str):
  """Decorator to register a runner implementation class.

  Args:
      name: The string identifier for the runner to attach to the registry.

  Returns:
      A class decorator function.
  """

  def decorator(cls: type[T]) -> type[T]:
    _REGISTRY[name] = cls
    return cls

  return decorator


def create_runner(
    name: base.RunnerType | str, config: dict[str, Any]
) -> base.AbstractRunner:
  """Creates a new instance of an runner from the registry.

  Args:
      name: The registered name of the runner to fetch.
      config: The initial configuration kwargs for the runner.

  Raises:
      ValueError: Upon fetching an unknown runner.

  Returns:
      An instantiated AbstractRunner runner.
  """
  _import_lazy_runner(name)
  if name not in _REGISTRY:
    raise ValueError(f"Runner '{name}' not found in registry.")
  cls = _REGISTRY[name]
  return typing.cast(base.AbstractRunner, cls(cls.config_type(**config)))


def get_runner_cls(name: base.RunnerType | str) -> type[_HasConfig]:
  """Returns the registered class of a runner."""
  _import_lazy_runner(name)
  if name not in _REGISTRY:
    raise ValueError(f"Runner '{name}' not found in registry.")
  return _REGISTRY[name]


def get_all_runners() -> list[str]:
  """Returns a list of all registered runner types.

  Returns:
      A list of string identifiers for all available runners.
  """
  return list(_REGISTRY) + [n for n in _LAZY_IMPORTS if n not in _REGISTRY]
