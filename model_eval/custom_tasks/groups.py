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

"""Group derivation layer for the custom evaluation framework."""

from typing import Iterable, List


def is_group(name: str, task_names: Iterable[str]) -> bool:
  """Determines if a string is a registered group prefix.

  A group is defined as any ':'-delimited prefix that is shared by one or more
  leaf tasks. An exact leaf task name is not considered a group.

  Args:
    name: The string prefix to check.
    task_names: The collection of all registered task names.

  Returns:
    True if the name represents a valid group, False otherwise.
  """
  return any(t.startswith(name + ":") for t in task_names)


def subtasks_of(name: str, task_names: Iterable[str]) -> List[str]:
  """Retrieves all leaf task names belonging to a specified group prefix.

  Args:
    name: The group prefix to expand.
    task_names: The collection of all registered task names.

  Returns:
    A sorted list of leaf task names matching the prefix.
  """
  return sorted(t for t in task_names if t.startswith(name + ":"))


def list_groups(task_names: Iterable[str]) -> List[str]:
  """Returns a sorted list of all derived task group prefixes.

  Args:
    task_names: The collection of all registered task names.

  Returns:
    A list of group prefix strings.
  """
  out = set()
  for t in task_names:
    parts = t.split(":")
    out.update(":".join(parts[:i]) for i in range(1, len(parts)))
  return sorted(out)
