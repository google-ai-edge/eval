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

"""Unit tests for dataset loader registry."""

from typing import Any, Iterable
from absl.testing import absltest
from model_eval import custom_tasks
from model_eval.custom_tasks.loaders import registry as loaders_registry


class LoadersRegistryTest(absltest.TestCase):
  """Unit tests verifying the behavior of the dataset loader registry."""

  def setUp(self):
    super().setUp()
    loaders_registry.clear_loaders()

  def tearDown(self):
    loaders_registry.clear_loaders()
    super().tearDown()

  def test_register_and_get(self):
    """Verifies that a loader can be successfully registered and retrieved."""

    def dummy_loader(spec: Any) -> Iterable[custom_tasks.DatasetRow]:
      del spec
      return []

    loaders_registry.register_loader("dummy", dummy_loader)
    resolved = loaders_registry.get_loader("dummy")
    self.assertIs(resolved, dummy_loader)
    self.assertEqual(list(resolved(None)), [])

  def test_duplicate_registration_raises(self):
    """Verifies that registering a loader with an existing name raises a ValueError."""

    def dup_loader(spec: Any) -> Iterable[custom_tasks.DatasetRow]:
      del spec
      return []

    loaders_registry.register_loader("dup", dup_loader)
    with self.assertRaises(ValueError):
      loaders_registry.register_loader("dup", dup_loader)

  def test_get_unknown_raises(self):
    """Verifies that resolving an unregistered loader name raises a KeyError."""
    with self.assertRaises(KeyError):
      loaders_registry.get_loader("nope")

  def test_list_loaders(self):
    """Verifies that list_loaders returns all correctly registered loader names."""

    def l1(spec: Any) -> Iterable[custom_tasks.DatasetRow]:
      del spec
      return []

    def l2(spec: Any) -> Iterable[custom_tasks.DatasetRow]:
      del spec
      return []

    loaders_registry.register_loader("list_l1", l1)
    loaders_registry.register_loader("list_l2", l2)
    self.assertIn("list_l1", loaders_registry.list_loaders())
    self.assertIn("list_l2", loaders_registry.list_loaders())

  def test_re_exports(self):
    """Verifies that primary loader registry functions are correctly re-exported."""
    self.assertIs(
        custom_tasks.register_loader, loaders_registry.register_loader
    )
    self.assertIs(custom_tasks.get_loader, loaders_registry.get_loader)
    self.assertIs(custom_tasks.list_loaders, loaders_registry.list_loaders)
    self.assertIs(custom_tasks.clear_loaders, loaders_registry.clear_loaders)


if __name__ == "__main__":
  absltest.main()
