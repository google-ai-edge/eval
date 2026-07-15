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
from model_eval.custom_tasks import groups


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
    new_name = task.name
    if new_name in self._custom:
      raise ValueError(f"Task '{new_name}' is already registered.")

    existing_names = list(self._custom.keys())

    if new_name in groups.list_groups(existing_names):
      raise ValueError(
          f"Cannot register leaf task '{new_name}' because it conflicts with an"
          " existing task group prefix."
      )

    parts = new_name.split(":")
    for i in range(1, len(parts)):
      prefix = ":".join(parts[:i])
      if prefix in self._custom:
        raise ValueError(
            f"Cannot register task '{new_name}' because prefix '{prefix}' is"
            " already registered as a leaf task."
        )

    self._custom[new_name] = task

  def get_task(self, name: str) -> base.CustomTask:
    """Retrieves a registered custom task by its name."""
    if name in self._custom:
      return self._custom[name]
    raise KeyError(f"Unknown custom task: '{name}'")

  def get_tasks(self, name: str) -> list[base.CustomTask]:
    """Resolves a name into a list of registered CustomTask instances.

    If name is an exact registered leaf task, returns a list containing just
    that task. If name is a derived task group prefix, returns all leaf tasks
    belonging to the group.

    Args:
      name: The task or group prefix name to resolve.

    Returns:
      A list of CustomTask instances.

    Raises:
      KeyError: If the name does not correspond to any registered task or group.
    """
    if name in self._custom:
      return [self._custom[name]]

    task_names = self.get_all_tasks()

    if groups.is_group(name, task_names):
      return [self._custom[t] for t in groups.subtasks_of(name, task_names)]

    raise KeyError(
        f"Unknown task or group: '{name}'; have {sorted(self._custom)}"
    )

  def get_all_tasks(self) -> list[str]:
    """Returns the names of all registered custom tasks."""
    return list(self._custom.keys())
