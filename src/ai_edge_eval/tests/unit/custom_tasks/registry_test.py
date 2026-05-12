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

"""Unit tests for TaskRegistry."""

from absl.testing import absltest
from ai_edge_eval import custom_tasks as tasks


class TaskRegistryTest(absltest.TestCase):

  def test_singleton_behavior(self):
    r1 = tasks.TaskRegistry.global_registry()
    r2 = tasks.TaskRegistry.global_registry()
    self.assertIs(r1, r2)

  def test_register_and_resolve(self):
    reg = tasks.TaskRegistry()
    task = tasks.CustomTask(
        name="foo", dataset="bar", metric_fn=lambda p, g: {}
    )
    reg.register(task)
    self.assertEqual(reg.get_task("foo"), task)

  def test_resolve_unknown_raises_keyerror(self):
    reg = tasks.TaskRegistry()
    with self.assertRaises(KeyError):
      reg.get_task("non_existent")

  def test_get_all_tasks(self):
    reg = tasks.TaskRegistry()
    task = tasks.CustomTask(
        name="foo", dataset="bar", metric_fn=lambda p, g: {}
    )
    reg.register(task)
    self.assertEqual(reg.get_all_tasks(), ["foo"])


if __name__ == "__main__":
  absltest.main()
