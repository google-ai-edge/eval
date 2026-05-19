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

"""Unit tests for base evaluation framework argument resolution."""

from typing import Any
import unittest
from model_eval.frameworks import base


class DummyFramework(base.AbstractEvalFramework):

  def evaluate(
      self, runner, tasks, limit=None, batch_size=None, eval_args=None
  ):
    pass

  @classmethod
  def supported_task_ids(cls) -> list[str]:
    return []

  @classmethod
  def supported_runners(cls) -> list[str]:
    return []

  @classmethod
  def describe_eval_args(cls) -> list[dict[str, Any]]:
    return []


class BaseFrameworkResolverTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.framework = DummyFramework()

  def test_explicit_priority_conflict(self):
    with self.assertRaises(ValueError):
      self.framework._from_unified_eval_args(
          limit=10, batch_size=None, eval_args={"limit": 20}
      )

  def test_dictionary_extraction(self):
    res = self.framework._from_unified_eval_args(
        limit=None, batch_size=None, eval_args={"limit": 20}
    )
    self.assertEqual(res.limit, 20)
    self.assertNotIn("limit", res.eval_args)

  def test_custom_key_mapping(self):
    res = self.framework._from_unified_eval_args(
        limit=None,
        batch_size=None,
        eval_args={"max_samples": 30},
        limit_key="max_samples",
    )
    self.assertEqual(res.limit, 30)
    self.assertNotIn("max_samples", res.eval_args)

  def test_defaulting(self):
    res = self.framework._from_unified_eval_args(
        limit=None, batch_size=None, eval_args={}, default_batch_size=1
    )
    self.assertEqual(res.batch_size, 1)

  def test_runner_args_extraction_and_injection(self):
    model_config = base.NativeModelConfig(
        model="test",
        model_path="/path/to/weights",
        device="cpu",
        model_args={"custom_arg": 123},
    )
    res = self.framework._from_unified_runner_args(
        model_config, model_path_key="weights_path"
    )
    self.assertEqual(res.model_args["weights_path"], "/path/to/weights")
    self.assertEqual(res.model_args["device"], "cpu")
    self.assertEqual(res.model_args["custom_arg"], 123)

  def test_runner_args_conflict(self):
    model_config = base.NativeModelConfig(
        model="test",
        model_path="/path/to/weights",
        model_args={"model_path": "/conflicting/path"},
    )
    with self.assertRaises(ValueError):
      self.framework._from_unified_runner_args(model_config)

  def test_runner_args_no_explicit_values(self):
    model_config = base.NativeModelConfig(
        model="test",
        model_args={"model_path": "/from/dict"},
    )
    res = self.framework._from_unified_runner_args(model_config)
    self.assertEqual(res.model_args["model_path"], "/from/dict")


if __name__ == "__main__":
  unittest.main()
