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

"""Loader registry for the custom evaluation framework."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, overload
from model_eval.custom_tasks import base


class LoaderFn(Protocol):
  """Protocol for registered dataset loader functions."""

  def __call__(self, spec: Any, /) -> Iterable[base.DatasetRow]:
    ...


_LOADERS: Dict[str, LoaderFn] = {}


@overload
def register_loader(name: str) -> Callable[[LoaderFn], LoaderFn]:
  ...


@overload
def register_loader(name: str, fn: LoaderFn) -> LoaderFn:
  ...


def register_loader(name: str, fn: Optional[LoaderFn] = None) -> Any:
  """Registers a dataset loader function under a unique name.

  Args:
    name: The string identifier for the loader.
    fn: Optional loader function to register. If provided, registers and returns
      it. If None, returns a decorator.

  Returns:
    The registered function, or a decorator that registers the function.

  Raises:
    ValueError: If the loader name is already registered.
  """

  def _reg(f: LoaderFn) -> LoaderFn:
    if name in _LOADERS:
      raise ValueError(f"loader '{name}' already registered")
    _LOADERS[name] = f
    return f

  return _reg(fn) if fn else _reg


def get_loader(name: str) -> LoaderFn:
  """Retrieves a registered dataset loader function by its name.

  Args:
    name: The name of the registered loader.

  Returns:
    The registered LoaderFn.

  Raises:
    KeyError: If the loader name is not found in the registry.
  """
  if name not in _LOADERS:
    raise KeyError(f"unknown loader '{name}'; have {sorted(_LOADERS)}")
  return _LOADERS[name]


def list_loaders() -> List[str]:
  """Returns a sorted list of all registered dataset loader names.

  Returns:
    A list of registered loader names.
  """
  return sorted(_LOADERS)


def clear_loaders() -> None:
  """Clears all registered loaders, restoring the registry to an empty state."""
  _LOADERS.clear()
