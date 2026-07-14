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

"""Unit tests for normalizer registry and identity normalizer."""

from absl.testing import absltest
from model_eval import custom_tasks
from model_eval.custom_tasks.metrics import normalize


class NormalizeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    normalize.clear_normalizers()

  def tearDown(self):
    normalize.clear_normalizers()
    super().tearDown()

  def test_identity(self):
    normalizer = normalize.get_normalizer(None)
    self.assertEqual(normalizer("A,b"), "A,b")

    normalizer_explicit = normalize.get_normalizer("identity")
    self.assertEqual(normalizer_explicit("A,b"), "A,b")

  def test_unknown_normalizer_raises(self):
    with self.assertRaises(KeyError):
      normalize.get_normalizer("xyz")

  def test_register_and_get_custom(self):
    def custom_norm(text):
      return text.lower()

    normalize.register_normalizer("lower", custom_norm)
    resolved_norm = normalize.get_normalizer("lower")
    self.assertEqual(resolved_norm("A,b"), "a,b")

  def test_duplicate_registration_raises(self):
    def norm(text):
      return text

    normalize.register_normalizer("dup_norm", norm)
    with self.assertRaises(ValueError):
      normalize.register_normalizer("dup_norm", norm)

  def test_list_normalizers(self):
    self.assertEqual(normalize.list_normalizers(), ["identity"])
    normalize.register_normalizer("dummy", lambda x: x)
    self.assertEqual(normalize.list_normalizers(), ["dummy", "identity"])

  def test_re_exports(self):
    self.assertIs(
        custom_tasks.register_normalizer, normalize.register_normalizer
    )
    self.assertIs(custom_tasks.get_normalizer, normalize.get_normalizer)
    self.assertIs(custom_tasks.list_normalizers, normalize.list_normalizers)
    self.assertIs(custom_tasks.clear_normalizers, normalize.clear_normalizers)


if __name__ == "__main__":
  absltest.main()
