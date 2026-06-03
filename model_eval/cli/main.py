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

"""ai_edge_eval CLI module."""

import importlib
import json
import pathlib
import sys
from typing import Any

from absl import app
from absl import flags
from model_eval.api import pipeline as eval_pipeline
from model_eval.frameworks import base as framework_base
from model_eval.runners import base as runners_base
from model_eval.runners import litert_lm
import click

# Maps recognized custom runner types to their corresponding runner
# configuration classes.
_CUSTOM_RUNNER_TYPES = {
    "litert-lm": litert_lm.LiteRtLmRunner.Config,
}

FLAGS = flags.FLAGS


def _parse_csv_args(csv_str: str) -> dict[str, Any]:
  """Parses a comma-separated key=value string into a dictionary, attempting JSON parsing."""
  res = {}
  if not csv_str:
    return res

  # We iterate character by character in order to distinguish between top-level
  # argument separator commas versus commas inside nested collections (such as
  # dict/list structures passed via JSON).
  items = []
  current = []
  depth = 0
  for char in csv_str:
    # Increment nesting depth when entering a dictionary or list.
    if char in "{[":
      depth += 1
    # Decrement nesting depth when closing a structure.
    elif char in "}]":
      depth -= 1

    # Split at commas only if we are at the top level (depth == 0).
    if char == "," and depth == 0:
      items.append("".join(current))
      current = []
    else:
      current.append(char)
  items.append("".join(current))

  for item in items:
    if "=" in item:
      key, value = item.split("=", 1)
      key, value = key.strip(), value.strip()
      try:
        # Normalize to JSON (convert single quotes to double quotes).
        res[key] = json.loads(value.replace("'", '"'))
      except json.JSONDecodeError:
        res[key] = value
  return res


def _build_model_config(
    runner_type: str,
    model_args: str,
    model_path: str | None = None,
    device: str | None = None,
) -> runners_base.RunnerConfig | framework_base.NativeModelConfig:
  """Constructs the runner or native model configuration from string arguments.

  Args:
      runner_type: Identifier of the runner or model to use.
      model_args: Comma-separated key=value string specifying model
        configuration parameters.
      model_path: Optional unified path argument to locate the model assets.
      device: Optional target device or hardware backend to run on.

  Returns:
      Either a fully instantiated runner Config or a NativeModelConfig object.
  """
  parsed_args = _parse_csv_args(model_args)
  if runner_type in _CUSTOM_RUNNER_TYPES:
    config_cls = _CUSTOM_RUNNER_TYPES[runner_type]
    return config_cls.from_unified_args(model_path, device, parsed_args)
  return framework_base.NativeModelConfig(
      model=runner_type,
      model_path=model_path,
      device=device,
      model_args=parsed_args,
  )


def _load_custom_tasks(custom_tasks_file: str | None) -> None:
  """Loads custom tasks from a python file."""
  if custom_tasks_file:
    path = pathlib.Path(custom_tasks_file)
    if str(path.parent.resolve()) not in sys.path:
      sys.path.insert(0, str(path.parent.resolve()))
    mod_name = path.stem
    importlib.import_module(mod_name)


def _execute_eval(
    runner,
    model_path,
    device,
    runner_args,
    tasks,
    limit,
    sample_range,
    batch_size,
    framework,
    custom_tasks_file,
    eval_args,
    output_dir,
    task_config,
    runner_config,
):
  """Executes the evaluation pipeline end-to-end via EvalPipeline.

  Args:
      runner: The string identifier of the model runner to use.
      model_path: Unified path identifying where the model resides.
      device: Requested execution hardware or backend device.
      runner_args: Captured string of extra configuration arguments for runner.
      tasks: Interable collection of benchmark task names to be ran.
      limit: Optional limit setting maximum samples or fraction per task.
      sample_range: Optional range of samples to evaluate (start,end).
      batch_size: Number of items processed at once during inference.
      framework: The framework name string used to structure evaluation.
      custom_tasks_file: Filepath mapping to a file registering custom tasks.
      eval_args: Key/value parameters influencing evaluation scoring.
      output_dir: Path to folder saving serialized outputs and sample records.
      task_config: Optional tasks.yaml override file path.
      runner_config: Optional runners.yaml override file path.
  """
  if not runner:
    raise click.UsageError("Missing option '--runner'.")
  if not tasks:
    raise click.UsageError("Missing option '--tasks'.")

  _load_custom_tasks(custom_tasks_file)

  model_config = _build_model_config(runner, runner_args, model_path, device)
  parsed_eval_args = _parse_csv_args(eval_args)

  # Initialize the EvalPipeline with parsed arguments.
  pipeline = eval_pipeline.EvalPipeline(
      model=model_config,
      framework=framework_base.FrameworkType(framework),
      tasks=tuple(tasks),
      limit=limit,
      sample_range=sample_range,
      batch_size=batch_size,
      output_dir=output_dir,
      task_config=task_config,
      runner_config=runner_config,
      eval_args=parsed_eval_args,
  )

  # Run the evaluation. Result generation and exporting is handled internally.
  results = pipeline.run()

  print()
  print(f"{'Task':40s} {'Metric':20s} {'Value'}")
  print("-" * 75)
  for task_name, metrics in results.aggregated_metrics.items():
    for metric_name, value in metrics.items():
      val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
      print(f"{task_name:40s} {metric_name:20s} {val_str}")


def _validate_sample_range(ctx, param, value):
  """Validates that sample_range start <= end."""
  if value and value[0] > value[1]:
    raise click.BadParameter(
        "Sample range start must be less than or equal to end."
    )
  return value


def _validate_limit(ctx, param, value):
  """Validates that limit is a float in (0, 1) or an integer >= 1."""
  if value is not None:
    try:
      fval = float(value)
      if fval <= 0:
        raise click.BadParameter("Limit must be greater than 0.")
      if fval < 1:
        return fval
      if not fval.is_integer():
        raise click.BadParameter("Limit >= 1 must be an integer.")
      return int(fval)
    except ValueError as e:
      raise click.BadParameter(f"Limit must be an integer or float. {e}") from e
  return None


@click.group(name="ai-edge-eval", invoke_without_command=True)
@click.version_option(version="0.0.1")
@click.option(
    "--runner", required=False, help="Runner type (e.g., litert-lm, hf)."
)
@click.option(
    "--model-path",
    default=None,
    help="Unified path identifier for the model.",
)
@click.option(
    "--device",
    default=None,
    help="Unified target execution device/backend.",
)
@click.option(
    "--runner-args",
    default="",
    help=(
        "Comma-separated key=value model arguments; value can be JSON-encoded"
        " (e.g., --runner-args model_path=/tmp/model.litertlm,backend=cpu)."
    ),
)
@click.option(
    "--tasks",
    multiple=True,
    required=False,
    help="Task names (e.g., --tasks ifeval --tasks mmlu).",
)
@click.option(
    "--limit",
    default=None,
    callback=_validate_limit,
    help="Maximum samples or fraction per task.",
)
@click.option(
    "--sample-range",
    default=None,
    type=(click.IntRange(min=0), click.IntRange(min=0)),
    callback=_validate_sample_range,
    help=(
        "Range of samples to evaluate, space-separated (e.g., --sample-range"
        " 0 10)."
    ),
)
@click.option(
    "--batch-size",
    default=1,
    type=int,
    help="Evaluation batch size.",
)
@click.option(
    "--framework",
    type=click.Choice(["lm-eval", "lighteval", "custom"]),
    default="lm-eval",
    help="Evaluation framework to use.",
)
@click.option(
    "--custom-tasks-file",
    default=None,
    help="Python file that registers custom tasks",
)
@click.option(
    "--eval-args",
    default="",
    help=(
        "Comma-separated key=value evaluation framework arguments; values can"
        " be JSON-encoded (e.g., --eval-args samples={'mmlu_astronomy': [0, 3,"
        " 6]},num_fewshot=5)."
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="/tmp/results/",
    help="Output directory.",
)
@click.option(
    "--task-config",
    type=click.Path(exists=True),
    default=None,
    help="Path to override bundled tasks.yaml.",
)
@click.option(
    "--runner-config",
    type=click.Path(exists=True),
    default=None,
    help="Path to override bundled runners.yaml.",
)
@click.pass_context
def cli(
    ctx,
    runner,
    model_path,
    device,
    runner_args,
    tasks,
    limit,
    sample_range,
    batch_size,
    framework,
    custom_tasks_file,
    eval_args,
    output_dir,
    task_config,
    runner_config,
):
  """Main entry point for the AI Edge Evaluation tool."""
  if ctx.invoked_subcommand is None:
    _execute_eval(
        runner,
        model_path,
        device,
        runner_args,
        tasks,
        limit,
        sample_range,
        batch_size,
        framework,
        custom_tasks_file,
        eval_args,
        output_dir,
        task_config,
        runner_config,
    )


@cli.command(name="eval")
@click.pass_context
def run_pipeline(ctx):
  """Runs the evaluation pipeline with the specified configuration."""
  parent_params = ctx.parent.params
  _execute_eval(
      runner=parent_params.get("runner"),
      model_path=parent_params.get("model_path"),
      device=parent_params.get("device"),
      runner_args=parent_params.get("runner_args"),
      tasks=parent_params.get("tasks"),
      limit=parent_params.get("limit"),
      sample_range=parent_params.get("sample_range"),
      batch_size=parent_params.get("batch_size"),
      framework=parent_params.get("framework"),
      custom_tasks_file=parent_params.get("custom_tasks_file"),
      eval_args=parent_params.get("eval_args"),
      output_dir=parent_params.get("output_dir"),
      task_config=parent_params.get("task_config"),
      runner_config=parent_params.get("runner_config"),
  )


@cli.command(name="list-tasks")
@click.option("--framework", default=None)
@click.option("--task-config", default=None)
@click.option("--custom-tasks-file", default=None)
def list_tasks(framework, task_config, custom_tasks_file):
  """Lists supported tasks for a given framework based on the allowlist."""
  _load_custom_tasks(custom_tasks_file)

  fw_enum = framework_base.FrameworkType(framework)
  allowed_tasks = eval_pipeline._load_task_allowlist(task_config, fw_enum)  # pylint: disable=protected-access

  click.echo(f"Supported tasks for framework '{framework}':")
  for task in sorted(allowed_tasks):
    click.echo(f"  - {task}")


@cli.command(name="list-runners")
@click.option("--framework", default=None)
@click.option("--runner-config", default=None)
def list_runners(framework, runner_config):
  """Lists supported runners for a given framework based on the allowlist."""

  fw_enum = framework_base.FrameworkType(framework)
  allowed_runners = eval_pipeline._load_runner_allowlist(runner_config, fw_enum)  # pylint: disable=protected-access

  click.echo(f"Supported runners for framework '{framework}':")
  for runner in sorted(allowed_runners):
    click.echo(f"  - {runner}")


@cli.command("list-args")
@click.option("--runner", default=None)
@click.option("--framework", default=None)
def list_args(runner, framework):
  if runner:
    click.echo(eval_pipeline.EvalPipeline.describe_runner_args(runner))
  if framework:
    click.echo(eval_pipeline.EvalPipeline.describe_eval_args(framework))
  if not runner and not framework:
    click.echo("Provide --runner and/or --framework.")


def main():
  """Entry point for the ai-edge-eval tool."""
  # We use a lambda flags_parser that ignores unknown flags (known_only=True)
  # to allow click to handle its own options without absl conflicts.
  app.run(_run_pipeline, flags_parser=lambda argv: FLAGS(argv, known_only=True))


def _run_pipeline(argv):
  """Internal main that receives argv from absl.app.run and starts Click."""
  # We pass the unparsed arguments to click.
  # argv[0] is the binary path, click expects argv[1:].
  try:
    # Using .main() allows passing specific arguments and handling exits
    cli.main(args=argv[1:], standalone_mode=False)
  except click.ClickException as e:
    e.show()
    sys.exit(e.exit_code)


if __name__ == "__main__":
  main()
