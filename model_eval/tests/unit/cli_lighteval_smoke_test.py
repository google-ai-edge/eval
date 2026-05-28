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

"""CLI subprocess smoke test for the Lighteval framework integration.

Verifies that the `ai-edge-eval list-tasks --framework lighteval` entry point
correctly loads and reports tasks from the internal allowlist, catching
import-order regressions and CLI registration bugs in fresh interpreter
instances.
"""

import os
import shutil
import subprocess

from absl.testing import absltest

try:
  from google3.third_party.bazel_rules.rules_python.python.runfiles import runfiles  # pylint: disable=g-import-not-at-top
except ImportError:
  runfiles = None


def _resolve_cli_bin() -> str:
  """Find the ai-edge-eval entry point via Blaze runfiles or pip install."""
  if runfiles:
    r = runfiles.Create()
    if r:
      found = r.Rlocation("google3/third_party/py/ai_edge_eval/cli")
      if found and os.path.exists(found):
        return found

  found = shutil.which("ai-edge-eval")
  if found and os.path.exists(found):
    return found

  raise absltest.SkipTest("ai-edge-eval CLI binary not found via Bazel or PATH")


class CliLightevalSmokeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.bin = _resolve_cli_bin()
    # Don't inherit a noisy parent env that could mask import errors.
    self.env = {**os.environ, "TRANSFORMERS_VERBOSITY": "error"}

  def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
    """Executes the `ai-edge-eval` CLI binary with arguments in an isolated subprocess.

    Captures stdout and stderr, enforces text-mode decoding, explicitly disables
    strict returncode checking (`check=False`), and applies a 60-second timeout.

    Args:
      *args: Command-line arguments and flags to forward to the executable.

    Returns:
      A `subprocess.CompletedProcess[str]` instance representing the finished
      invocation.
    """
    return subprocess.run(
        [self.bin, *args],
        capture_output=True,
        text=True,
        env=self.env,
        timeout=60,
        check=False,
    )

  def test_list_tasks_lighteval(self):
    """list-tasks --framework lighteval returns the yaml allowlist and exits 0."""
    p = self._run("list-tasks", "--framework", "lighteval")
    self.assertEqual(p.returncode, 0, msg=p.stderr)
    out = p.stdout
    # Known tasks from model_eval/config/tasks.yaml (lighteval section).
    for task in ("arc:easy", "arc:challenge", "mmlu", "winogrande", "ifeval"):
      self.assertIn(
          task, out, msg=f"task {task!r} missing from list-tasks output"
      )

  def test_list_runners_lighteval(self):
    """list-runners --framework lighteval includes both runner paths and exits 0."""
    p = self._run("list-runners", "--framework", "lighteval")
    self.assertEqual(p.returncode, 0, msg=p.stderr)
    out = p.stdout
    for runner in ("litert-lm", "accelerate"):
      self.assertIn(
          runner, out, msg=f"runner {runner!r} missing from list-runners output"
      )

  def test_list_args_lighteval(self):
    """list-args --framework lighteval surfaces lighteval PipelineParameters fields."""
    p = self._run("list-args", "--framework", "lighteval")
    self.assertEqual(p.returncode, 0, msg=p.stderr)
    out = p.stdout
    # 'max_samples' is the field the adapter maps --limit onto; if this is
    # missing, the introspection path is broken.
    self.assertIn("max_samples", out)

  def test_unknown_framework_is_handled(self):
    """An unknown --framework value exits non-zero with a clear error.

    This guards against silent fall-through that could mask a real
    misregistration of the lighteval framework type.
    """
    p = self._run("list-tasks", "--framework", "definitely-not-a-framework")
    self.assertNotEqual(p.returncode, 0)


if __name__ == "__main__":
  absltest.main()
