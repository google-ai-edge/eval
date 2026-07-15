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

"""Unit tests verifying CLI loading of Python registrars and YAML catalogs."""

import pathlib
import tempfile
from absl.testing import absltest
from model_eval import custom_tasks
from model_eval.cli import main as cli_main


class CliYamlTest(absltest.TestCase):
  """Unit tests for multi-file --custom-tasks-file CLI execution."""

  def setUp(self):
    """Resets the global TaskRegistry and dummy registries."""
    super().setUp()
    custom_tasks.TaskRegistry.global_registry()._custom.clear()
    if "yaml_cli_loader" not in custom_tasks.list_loaders():
      custom_tasks.register_loader("yaml_cli_loader", lambda spec: iter([]))
    if "yaml_cli_metric" not in custom_tasks.list_metrics():
      custom_tasks.register_metric(
          "yaml_cli_metric", lambda p, g, r, *, normalizer: {}
      )
    self.temp_dir = tempfile.TemporaryDirectory()
    self.temp_path = pathlib.Path(self.temp_dir.name)

  def tearDown(self):
    super().tearDown()
    self.temp_dir.cleanup()

  def test_cli_yaml_registers_tasks(self):
    """Verifies that passing a YAML catalog path correctly registers tasks."""
    yaml_file = self.temp_path / "byo.yaml"
    yaml_file.write_text(
        "- {name: t1, loader: yaml_cli_loader, metrics: [yaml_cli_metric],"
        " generation_config: {}}\n"
    )
    cli_main._load_custom_tasks([str(yaml_file)])
    self.assertIn(
        "t1", custom_tasks.TaskRegistry.global_registry().get_all_tasks()
    )

  def test_cli_python_file_still_works(self):
    """Verifies that passing an existing Python registrar script functions flawlessly."""
    py_file = self.temp_path / "my_unique_eval_registrar_584.py"
    py_file.write_text(
        "from model_eval import custom_tasks\n"
        "task = custom_tasks.CustomTask(\n"
        "    name='py_task',\n"
        "    dataset=lambda: iter([]),\n"
        "    metric_fn=lambda p, g, r: {},\n"
        ")\n"
        "custom_tasks.TaskRegistry.global_registry().register(task)\n"
    )
    cli_main._load_custom_tasks([str(py_file)])
    self.assertIn(
        "py_task", custom_tasks.TaskRegistry.global_registry().get_all_tasks()
    )

  def test_py_loaded_before_yaml_regardless_of_flag_order(self):
    """Verifies that Python files are loaded before YAML files, regardless of CLI argument order."""
    plug_file = self.temp_path / "my_unique_eval_plugins_584.py"
    plug_file.write_text(
        "from model_eval import custom_tasks\n"
        "if 'custom_m' not in custom_tasks.list_metrics():\n"
        "  custom_tasks.register_metric('custom_m', lambda p, g, r, *,"
        " normalizer: {'m': 0})\n"
    )
    cat_file = self.temp_path / "catalog.yaml"
    cat_file.write_text(
        "- {name: t2, loader: yaml_cli_loader, metrics: [custom_m],"
        " generation_config: {}}\n"
    )
    # Pass YAML first, Python second
    cli_main._load_custom_tasks([str(cat_file), str(plug_file)])
    self.assertIn(
        "t2", custom_tasks.TaskRegistry.global_registry().get_all_tasks()
    )


if __name__ == "__main__":
  absltest.main()
