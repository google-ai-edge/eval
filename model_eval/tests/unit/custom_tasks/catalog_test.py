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

"""Unit tests for declarative TaskSpec and catalog logic."""

import pathlib
import tempfile
from absl.testing import absltest
from model_eval import custom_tasks
from model_eval.config import generation_config
from model_eval.custom_tasks import catalog


class CatalogTest(absltest.TestCase):
  """Unit tests verifying TaskSpec, YAML catalog loading, and task building."""

  def setUp(self):
    """Initializes dummy metric and loader registrations for validation."""
    super().setUp()
    self.dummy_row = custom_tasks.DatasetRow(
        messages=[],
        ground_truth="target",
    )
    if "catalog_dummy_loader" not in custom_tasks.list_loaders():
      custom_tasks.register_loader(
          "catalog_dummy_loader", lambda spec: iter([self.dummy_row])
      )
    if "catalog_dummy_metric" not in custom_tasks.list_metrics():
      custom_tasks.register_metric(
          "catalog_dummy_metric", lambda p, g, r, *, normalizer: {"dummy": 1.0}
      )
    self.temp_dir = tempfile.TemporaryDirectory()
    self.temp_path = pathlib.Path(self.temp_dir.name)

  def tearDown(self):
    super().tearDown()
    self.temp_dir.cleanup()

  def test_for_each_expands_and_renders(self):
    """Verifies that expand_for_each correctly computes Cartesian matrix permutations."""
    raw = {
        "name": "task_{x}_{y}",
        "loader": "catalog_dummy_loader",
        "metrics": ["catalog_dummy_metric"],
        "for_each": {"x": ["a", "b"], "y": [1, 2]},
    }
    expanded = catalog.expand_for_each(raw)
    names = [spec["name"] for spec in expanded]
    self.assertEqual(names, ["task_a_1", "task_a_2", "task_b_1", "task_b_2"])

  def test_for_each_absent_returns_single(self):
    """Verifies that expand_for_each returns the unmutated entry when for_each is absent."""
    raw = {
        "name": "task_single",
        "loader": "catalog_dummy_loader",
        "metrics": ["catalog_dummy_metric"],
    }
    expanded = catalog.expand_for_each(raw)
    self.assertLen(expanded, 1)
    self.assertEqual(expanded[0]["name"], "task_single")

  def test_unknown_loader_or_metric_rejected(self):
    """Verifies that TaskSpec validation correctly rejects unregistered names."""
    with self.assertRaises(ValueError):
      catalog.TaskSpec(
          name="bad_loader",
          loader="unknown_loader",
          metrics=["catalog_dummy_metric"],
          generation_config=generation_config.GenerationConfig(),
      )
    with self.assertRaises(ValueError):
      catalog.TaskSpec(
          name="bad_metric",
          loader="catalog_dummy_loader",
          metrics=["unknown_metric"],
          generation_config=generation_config.GenerationConfig(),
      )

  def test_load_catalog_and_dupes(self):
    """Verifies that load_catalog loads valid files and correctly rejects duplicates."""
    valid_file = self.temp_path / "valid.yaml"
    valid_file.write_text(
        "- {name: t1, loader: catalog_dummy_loader, metrics:"
        " [catalog_dummy_metric], generation_config: {}}\n- {name: t2, loader:"
        " catalog_dummy_loader, metrics: [catalog_dummy_metric],"
        " generation_config: {}}\n"
    )
    specs = catalog.load_catalog(str(valid_file))
    self.assertEqual({s.name for s in specs}, {"t1", "t2"})

    dup_file = self.temp_path / "dup.yaml"
    dup_file.write_text(
        "- {name: t1, loader: catalog_dummy_loader, metrics:"
        " [catalog_dummy_metric], generation_config: {}}\n- {name: t1, loader:"
        " catalog_dummy_loader, metrics: [catalog_dummy_metric],"
        " generation_config: {}}\n"
    )
    with self.assertRaises(ValueError):
      catalog.load_catalog(str(dup_file))

  def test_build_task_wires_dataset_and_metric(self):
    """Verifies that build_task correctly resolves a TaskSpec into a CustomTask instance."""
    spec = catalog.TaskSpec(
        name="concrete_task",
        loader="catalog_dummy_loader",
        metrics=["catalog_dummy_metric"],
        generation_config=generation_config.GenerationConfig(
            max_new_tokens=128),
    )
    task = catalog.build_task(spec)
    self.assertEqual(task.name, "concrete_task")
    self.assertEqual(task.generation_config.max_new_tokens, 128)
    self.assertEqual(task.generation_config.temperature, 1.0)

    # Verify lazy dataset execution returns the dummy row
    rows = list(task.dataset())
    self.assertEqual(rows, [self.dummy_row])

    # Verify composed metric execution
    res = task.metric_fn(["pred"], ["gt"], [self.dummy_row])
    self.assertEqual(res, {"dummy": 1.0})

  def test_build_task_with_generation_config(self):
    """Verifies that build_task respects generation_config from TaskSpec."""
    spec = catalog.TaskSpec(
        name="concrete_task",
        loader="catalog_dummy_loader",
        metrics=["catalog_dummy_metric"],
        generation_config=generation_config.GenerationConfig(
            temperature=0.5,
            max_new_tokens=100,
            stop_sequences=["\n"],
        ),
    )
    task = catalog.build_task(spec)
    self.assertEqual(task.name, "concrete_task")
    self.assertEqual(task.generation_config.temperature, 0.5)
    self.assertEqual(task.generation_config.max_new_tokens, 100)
    self.assertEqual(task.generation_config.stop_sequences, ["\n"])

  def test_build_task_generation_config_defaults(self):
    """Verifies default values when generation_config is partially set."""
    spec = catalog.TaskSpec(
        name="concrete_task",
        loader="catalog_dummy_loader",
        metrics=["catalog_dummy_metric"],
        generation_config=generation_config.GenerationConfig(),
    )
    task = catalog.build_task(spec)
    self.assertEqual(task.generation_config.temperature, 1.0)
    self.assertEqual(task.generation_config.max_new_tokens, 256)

  def test_register_specs_populates_registry(self):
    """Verifies that register_specs successfully registers concrete tasks into the registry."""
    spec = catalog.TaskSpec(
        name="reg_task",
        loader="catalog_dummy_loader",
        metrics=["catalog_dummy_metric"],
        generation_config=generation_config.GenerationConfig(),
    )
    # Use global registry for test
    reg = custom_tasks.TaskRegistry.global_registry()
    catalog.register_specs([spec], registry_instance=reg)
    self.assertIn("reg_task", reg.get_all_tasks())


if __name__ == "__main__":
  absltest.main()
