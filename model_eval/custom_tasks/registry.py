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

"""Registry for custom evaluation tasks."""

from typing import Optional

from model_eval.custom_tasks import base


class TaskRegistry:
  """Singleton registry for managing CustomTask definitions."""

  _instance: Optional["TaskRegistry"] = None

  def __init__(self):
    """Initializes an empty custom task registry."""
    self._custom: dict[str, base.CustomTask] = {}

  @classmethod
  def global_registry(cls) -> "TaskRegistry":
    """Returns the singleton instance of the TaskRegistry."""
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def register(self, task: base.CustomTask) -> None:
    """Registers a new custom task in the registry by name."""
    self._custom[task.name] = task

  def get_task(self, name: str) -> base.CustomTask:
    """Retrieves a registered custom task by its name."""
    if name in self._custom:
      return self._custom[name]
    raise KeyError(f"Unknown custom task: '{name}'")

  def get_all_tasks(self) -> list[str]:
    """Returns the names of all registered custom tasks."""
    return list(self._custom.keys())
