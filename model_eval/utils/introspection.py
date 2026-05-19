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

"""Utility for component field introspection without layer coupling."""

import dataclasses
import inspect
from typing import Any

import pydantic


def get_fields(source) -> list[dict[str, Any]]:
  """Extracts field metadata from a BaseModel, dataclass, or function signature."""
  if source is None:
    return []

  # If the source is a callable that takes no arguments, we call it
  # to get the underlying object to inspect.
  if callable(source) and not isinstance(source, type):
    try:
      sig = inspect.signature(source)
      if not sig.parameters:
        source = source()
    except (TypeError, ValueError):
      pass

  # Handle Pydantic BaseModels: extract model fields and their requirements.
  if isinstance(source, type) and issubclass(source, pydantic.BaseModel):
    return [
        {
            "name": n,
            "type": str(f.annotation),
            "default": "required" if f.is_required() else f.default,
        }
        for n, f in source.model_fields.items()
    ]

  # Handle standard Python dataclasses.
  if dataclasses.is_dataclass(source):
    return [
        {
            "name": f.name,
            "type": str(f.type),
            "default": (
                "required" if f.default is dataclasses.MISSING else f.default
            ),
        }
        for f in dataclasses.fields(source)
    ]

  # Fallback: inspect the signature of a general callable or class constructor.
  sig = inspect.signature(source if callable(source) else source.__init__)
  return [
      {
          "name": n,
          "type": str(p.annotation),
          "default": (
              "required" if p.default is inspect.Parameter.empty else p.default
          ),
      }
      for n, p in sig.parameters.items()
      if n not in ("self", "kwargs")
  ]
