import argparse
import datetime
import os
import re
import subprocess
import sys

# Mapping of base package name to its nightly package name for nightly build.
NIGHTLY_DEPENDENCIES = {
    "litert-lm": "litert-lm-nightly",
}


def configure_build(is_nightly: bool, restore: bool = False):
  pyproject_path = "pyproject.toml"

  if not os.path.exists(pyproject_path):
    raise FileNotFoundError(
        "Could not find pyproject.toml. Please run this script from"
        " the repository root."
    )

  with open(pyproject_path, "r") as f:
    content = f.read()

  # 1. Handle Project Name
  # Normalize project name to base 'ai-edge-eval'
  content = content.replace(
      'name = "ai-edge-eval-nightly"', 'name = "ai-edge-eval"'
  )
  content = content.replace('"ai-edge-eval-nightly[', '"ai-edge-eval[')

  if is_nightly and not restore:
    content = content.replace(
        'name = "ai-edge-eval"', 'name = "ai-edge-eval-nightly"'
    )
    content = content.replace('"ai-edge-eval[', '"ai-edge-eval-nightly[')

  # 2. Handle Version
  # Find version in pyproject.toml
  version_match = re.search(r'version = "([^"]*)"', content)
  if not version_match:
    raise ValueError("Could not find version in pyproject.toml")

  original_version = version_match.group(1)
  base_version = original_version.split(".dev")[0]

  if restore or not is_nightly:
    # Restore or stable build uses clean base version
    content = re.sub(
        r'version = "[^"]*"', f'version = "{base_version}"', content
    )
  else:
    # Nightly build appends .devYYYYMMDD
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    nightly_version = f"{base_version}.dev{date_str}"
    content = re.sub(
        r'version = "[^"]*"', f'version = "{nightly_version}"', content
    )
    print(f"Configured nightly version: {nightly_version}")

  # 3. Configure dependencies
  # For ai_edge_eval, we want to replace "litert-lm" with "litert-lm-nightly" in nightly builds
  to_nightly_deps = is_nightly and not restore

  for stable_dep, nightly_dep in NIGHTLY_DEPENDENCIES.items():
    if to_nightly_deps:
      # Replace "litert-lm" with "litert-lm-nightly"
      content = content.replace(f'"{stable_dep}"', f'"{nightly_dep}"')
    else:
      # Restore "litert-lm-nightly" with "litert-lm"
      content = content.replace(f'"{nightly_dep}"', f'"{stable_dep}"')

  with open(pyproject_path, "w") as f:
    f.write(content)


def main():
  parser = argparse.ArgumentParser(
      description="Build stable or nightly wheels using uv."
  )
  parser.add_argument(
      "--type",
      choices=["stable", "nightly"],
      default="nightly",
      help="Type of wheel to build (stable or nightly).",
  )
  args = parser.parse_args()

  is_nightly = args.type == "nightly"
  print(f"Configuring setup for {args.type} build...")
  configure_build(is_nightly=is_nightly, restore=False)

  try:
    print("Running 'uv build'...")
    subprocess.run(["uv", "build"], check=True)
    print(f"Successfully built {args.type} wheels in ./dist")
  except subprocess.CalledProcessError as e:
    print(f"Error during build: {e}", file=sys.stderr)
    sys.exit(1)
  finally:
    # Always restore files back to development defaults
    print("Restoring files to development defaults...")
    configure_build(is_nightly=is_nightly, restore=True)


if __name__ == "__main__":
  main()
