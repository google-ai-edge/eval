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

"""Unit tests for metric registry and compose."""

from absl.testing import absltest
from model_eval import custom_tasks
from model_eval.custom_tasks.metrics import registry as metrics_registry


class MetricsRegistryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    metrics_registry.clear_metrics()

  def tearDown(self):
    metrics_registry.clear_metrics()
    super().tearDown()

  def test_register_and_get(self):

    def dummy_metric(*args, **kwargs):
      del args, kwargs
      return {"dummy": 1.0}

    metrics_registry.register_metric("dummy", dummy_metric)
    resolved_metric = metrics_registry.get_metric("dummy")
    self.assertEqual(
        resolved_metric(
            ["a"],
            ["a"],
            [{"messages": [], "ground_truth": "a"}],
            normalizer=lambda x: x,
        ),
        {"dummy": 1.0},
    )

  def test_duplicate_registration_raises(self):

    def dup_metric(*args, **kwargs):
      del args, kwargs
      return {}

    metrics_registry.register_metric("dup", dup_metric)
    with self.assertRaises(ValueError):
      metrics_registry.register_metric("dup", dup_metric)

  def test_get_unknown_raises(self):
    with self.assertRaises(KeyError):
      metrics_registry.get_metric("nope")

  def test_list_metrics(self):

    def m1(*args, **kwargs):
      del args, kwargs
      return {}

    def m2(*args, **kwargs):
      del args, kwargs
      return {}

    metrics_registry.register_metric("list_m1", m1)
    metrics_registry.register_metric("list_m2", m2)
    self.assertIn("list_m1", metrics_registry.list_metrics())
    self.assertIn("list_m2", metrics_registry.list_metrics())

  def test_compose_empty_raises(self):
    with self.assertRaises(ValueError):
      metrics_registry.compose([])

  def test_compose_merges(self):
    metrics_registry.register_metric(
        "comp_m1", lambda *args, **kwargs: {"m1": 1.0}
    )
    metrics_registry.register_metric(
        "comp_m2", lambda *args, **kwargs: {"m2": 2.0}
    )

    composed_fn = metrics_registry.compose(["comp_m1", "comp_m2"])
    self.assertEqual(
        composed_fn(["a"], ["a"], [{"messages": [], "ground_truth": "a"}]),
        {"m1": 1.0, "m2": 2.0},
    )

  def test_compose_collision_raises(self):
    metrics_registry.register_metric(
        "collision_m1", lambda *args, **kwargs: {"score": 1.0}
    )
    metrics_registry.register_metric(
        "collision_m2", lambda *args, **kwargs: {"score": 2.0}
    )

    composed_fn = metrics_registry.compose(["collision_m1", "collision_m2"])
    with self.assertRaises(ValueError):
      composed_fn(["a"], ["a"], [{"messages": [], "ground_truth": "a"}])

  def test_re_exports(self):
    self.assertIs(
        custom_tasks.register_metric, metrics_registry.register_metric
    )
    self.assertIs(custom_tasks.get_metric, metrics_registry.get_metric)
    self.assertIs(custom_tasks.list_metrics, metrics_registry.list_metrics)
    self.assertIs(custom_tasks.clear_metrics, metrics_registry.clear_metrics)
    self.assertIs(custom_tasks.compose, metrics_registry.compose)


if __name__ == "__main__":
  absltest.main()
