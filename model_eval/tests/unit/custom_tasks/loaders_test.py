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

"""Unit tests for loaders and slicing."""

from absl.testing import absltest
from model_eval.custom_tasks import loaders


class LoadersTest(absltest.TestCase):

  def test_parse_samples_hyphen(self):
    self.assertEqual(loaders.parse_samples("1-3", 10), [1, 2, 3])

  def test_parse_samples_colon_step(self):
    self.assertEqual(loaders.parse_samples("0:5:2", 10), [0, 2, 4])

  def test_parse_samples_comma(self):
    self.assertEqual(loaders.parse_samples("0,2,4", 10), [0, 2, 4])


if __name__ == "__main__":
  absltest.main()
