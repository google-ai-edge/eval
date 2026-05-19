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

"""Unit tests for introspection utility."""

import dataclasses
import unittest
from model_eval.utils import introspection
import pydantic


@dataclasses.dataclass
class DummyDataClass:
  a: int
  b: str = "default"


class DummyBaseModel(pydantic.BaseModel):
  x: int
  y: str = "test"


def dummy_function(param1: int, param2: str = "val"):
  pass


class TestIntrospection(unittest.TestCase):

  def test_get_fields_dataclass(self):
    fields = introspection.get_fields(DummyDataClass)
    self.assertEqual(len(fields), 2)
    self.assertEqual(fields[0]["name"], "a")
    self.assertEqual(fields[0]["default"], "required")
    self.assertEqual(fields[1]["name"], "b")
    self.assertEqual(fields[1]["default"], "default")

  def test_get_fields_basemodel(self):
    fields = introspection.get_fields(DummyBaseModel)
    self.assertEqual(len(fields), 2)
    self.assertEqual(fields[0]["name"], "x")
    self.assertEqual(fields[0]["default"], "required")
    self.assertEqual(fields[1]["name"], "y")
    self.assertEqual(fields[1]["default"], "test")

  def test_get_fields_function(self):
    fields = introspection.get_fields(dummy_function)
    self.assertEqual(len(fields), 2)
    self.assertEqual(fields[0]["name"], "param1")
    self.assertEqual(fields[0]["default"], "required")
    self.assertEqual(fields[1]["name"], "param2")
    self.assertEqual(fields[1]["default"], "val")


if __name__ == "__main__":
  unittest.main()
