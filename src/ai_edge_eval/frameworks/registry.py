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

"""Registry for framework implementations."""

from ai_edge_eval.frameworks import base

_REGISTRY: dict[str, type[base.AbstractEvalFramework]] = {}


def register_framework(name: base.FrameworkType | str):
  """Decorator to register a framework implementation class.

  Args:
      name: The string identifier for the framework to attach to the registry.

  Returns:
      A class decorator function.
  """

  def decorator(cls: type[base.AbstractEvalFramework]):
    _REGISTRY[name] = cls
    return cls

  return decorator


def get_framework(name: base.FrameworkType | str) -> base.AbstractEvalFramework:
  """Retrieves a new instance of an evaluation framework from the registry.

  Args:
      name: The registered name of the framework to fetch.

  Raises:
      ValueError: Upon fetching an unknown framework.

  Returns:
      An instantiated AbstractEvalFramework framework.
  """
  if name not in _REGISTRY:
    raise ValueError(f"Framework '{name}' not found in registry.")
  return _REGISTRY[name]()  # type: ignore
