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

"""Unit tests for CLI discovery commands."""

from absl.testing import absltest
from model_eval import custom_tasks
from model_eval.cli import main
from click import testing


class CliDiscoveryTest(absltest.TestCase):
  """Unit tests verifying the custom framework CLI discovery subcommands."""

  def setUp(self):
    super().setUp()
    custom_tasks.clear_metrics()
    custom_tasks.clear_normalizers()
    custom_tasks.clear_loaders()

  def tearDown(self):
    custom_tasks.clear_metrics()
    custom_tasks.clear_normalizers()
    custom_tasks.clear_loaders()
    super().tearDown()

  def test_list_metrics_cmd(self):
    """Verifies that the list-metrics CLI command correctly lists registered metrics."""

    def m(*args, **kwargs):
      del args, kwargs
      return {}

    custom_tasks.register_metric("discovery_test_metric", m)

    runner = testing.CliRunner()
    result = runner.invoke(main.cli, ["list-metrics"])
    self.assertEqual(result.exit_code, 0)
    self.assertIn("discovery_test_metric", result.output)

  def test_list_loaders_cmd(self):
    """Verifies that the list-loaders CLI command correctly lists registered loaders."""

    def l(spec):
      del spec
      return []

    custom_tasks.register_loader("discovery_test_loader", l)

    runner = testing.CliRunner()
    result = runner.invoke(main.cli, ["list-loaders"])
    self.assertEqual(result.exit_code, 0)
    self.assertIn("discovery_test_loader", result.output)


if __name__ == "__main__":
  absltest.main()
