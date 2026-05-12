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

"""Unit tests for GenerationConfig."""

from absl.testing import absltest
from ai_edge_eval import config
import pydantic


class GenerationConfigTest(absltest.TestCase):

  def test_default_construction(self):
    cfg = config.GenerationConfig()
    self.assertEqual(cfg.temperature, 1.0)
    self.assertEqual(cfg.max_new_tokens, 256)
    self.assertEqual(cfg.stop_sequences, [])

  def test_custom_construction(self):
    cfg = config.GenerationConfig(
        temperature=0.7, max_new_tokens=128, stop_sequences=["\n"]
    )
    self.assertEqual(cfg.temperature, 0.7)
    self.assertEqual(cfg.max_new_tokens, 128)
    self.assertEqual(cfg.stop_sequences, ["\n"])

  def test_validation_error(self):
    with self.assertRaises(pydantic.ValidationError):
      config.GenerationConfig(temperature="not-a-float")
    with self.assertRaises(pydantic.ValidationError):
      config.GenerationConfig(max_new_tokens="not-an-int")


if __name__ == "__main__":
  absltest.main()
