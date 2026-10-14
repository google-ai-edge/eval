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

"""Normalizer registry and identity normalizer."""

from typing import Any, Callable, Dict, List, Optional, Union, overload

Normalizer = Callable[[str], str]

_NORMALIZERS: Dict[str, Normalizer] = {}


@overload
def register_normalizer(name: str) -> Callable[[Normalizer], Normalizer]:
  ...


@overload
def register_normalizer(name: str, fn: Normalizer) -> Normalizer:
  ...


def register_normalizer(name: str, fn: Optional[Normalizer] = None) -> Any:
  """Registers a text normalizer function under a unique name.

  Args:
    name: The string identifier for the normalizer.
    fn: Optional normalizer function to register. If provided, registers and
      returns it. If None, returns a decorator.

  Returns:
    The registered function, or a decorator that registers the function.

  Raises:
    ValueError: If the normalizer name is already registered.
  """

  def _reg(f: Normalizer) -> Normalizer:
    if name in _NORMALIZERS:
      raise ValueError(f"normalizer '{name}' already registered")
    _NORMALIZERS[name] = f
    return f

  if fn is not None:
    return _reg(fn)
  return _reg


def get_normalizer(name: Optional[str]) -> Normalizer:
  """Retrieves a registered text normalizer function by its name.

  Args:
    name: The name of the registered normalizer. If None, returns the default
      identity normalizer.

  Returns:
    The registered Normalizer function.

  Raises:
    KeyError: If the normalizer name is not found in the registry.
  """
  if name is None:
    return identity
  if name not in _NORMALIZERS:
    raise KeyError(f"unknown normalizer '{name}'; have {sorted(_NORMALIZERS)}")
  return _NORMALIZERS[name]


def list_normalizers() -> List[str]:
  """Returns a sorted list of all registered normalizer names.

  Returns:
    A list of registered normalizer names.
  """
  return sorted(_NORMALIZERS)


def clear_normalizers() -> None:
  """Clears all registered normalizers, restoring to the built-in identity state."""
  _NORMALIZERS.clear()
  _NORMALIZERS["identity"] = identity


@register_normalizer("identity")
def identity(text: str) -> str:
  """Default identity normalizer that returns text unchanged.

  Args:
    text: Input text string.

  Returns:
    The unmodified input text.
  """
  return text
