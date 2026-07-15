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

"""Unit tests for prefix-based custom task groups."""

from absl.testing import absltest
from model_eval.custom_tasks import groups


class GroupsTest(absltest.TestCase):
  """Unit tests verifying prefix-derived task groups functionality."""

  def test_prefix_based_groups(self):
    """Verifies that groups are correctly identified and resolved from namespaced strings."""
    names = ["qa:squad:dev", "qa:squad:test", "qa:nq:dev"]

    self.assertTrue(groups.is_group("qa", names))
    self.assertTrue(groups.is_group("qa:squad", names))
    self.assertTrue(groups.is_group("qa:nq", names))

    # Leaf task names are not groups
    self.assertFalse(groups.is_group("qa:squad:dev", names))
    self.assertFalse(groups.is_group("unknown_group", names))

    self.assertEqual(
        groups.subtasks_of("qa:squad", names), ["qa:squad:dev", "qa:squad:test"]
    )
    self.assertEqual(
        groups.subtasks_of("qa", names),
        ["qa:nq:dev", "qa:squad:dev", "qa:squad:test"],
    )

    self.assertTrue(
        {"qa", "qa:nq", "qa:squad"}.issubset(set(groups.list_groups(names)))
    )


if __name__ == "__main__":
  absltest.main()
