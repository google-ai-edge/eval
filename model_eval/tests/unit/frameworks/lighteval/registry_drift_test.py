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

"""Drift tests for the Lighteval registry vs. our tasks.yaml allowlist."""

import os
from absl.testing import absltest
from model_eval.frameworks.lighteval import lighteval
import yaml


class TestRegistryDrift(absltest.TestCase):
  """Ensures the Lighteval registry matches our internal tasks.yaml allowlist."""

  def test_supported_task_ids_is_nonempty(self):
    ids = lighteval.LightEvalFramework.supported_task_ids()
    self.assertNotEmpty(ids)

  def test_suite_aliases_are_included(self):
    """Explicit guard: known lighteval suites (mmlu, bigbench_hard) must be reported.

    Lighteval expands suite names at invocation time but the adapter's task
    allowlist consumer needs to know they are valid. This test ensures the
    adapter unions the registry's _task_superset_dict with the concrete task
    map.
    """
    supported = set(lighteval.LightEvalFramework.supported_task_ids())
    for suite in ("mmlu", "bigbench_hard"):
      self.assertIn(
          suite,
          supported,
          msg=(
              f"task suite alias {suite!r} missing from "
              "supported_task_ids(). The adapter is no longer unioning the "
              "registry's _task_superset_dict with the concrete task map."
          ),
      )

  def test_every_yaml_task_is_reported_by_supported_task_ids(self):
    """Every task in tasks.yaml lighteval section is in supported_task_ids()."""
    config_path = os.path.join(
        absltest.get_default_test_srcdir(),
        "model_eval/config/tasks.yaml",
    )
    if not os.path.exists(config_path):
      self.skipTest(f"config/tasks.yaml not found at {config_path}")

    with open(config_path, "r") as f:
      config = yaml.safe_load(f)

    yaml_tasks = config.get("lighteval", [])
    supported = set(lighteval.LightEvalFramework.supported_task_ids())
    for t in yaml_tasks:
      self.assertIn(
          t,
          supported,
          msg=f"Task {t!r} in tasks.yaml not reported by supported_task_ids()",
      )


if __name__ == "__main__":
  absltest.main()

