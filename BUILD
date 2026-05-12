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

load("//devtools/copybara/rules:copybara.bzl", "copybara_config_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")
load("//third_party/py/etils:build_defs.bzl", "glob_py_srcs")
load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = [":internal"],
)

licenses(["notice"])

# new-style license rule. This one has a default attribute that links to the LICENSE file.
license(name = "license")

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//third_party/py/ai_edge_eval/...",
    ],
)

# ai_edge_eval public API.
# This pytype_strict_library rule centralizes all files/deps.
pytype_strict_library(
    name = "ai_edge_eval",
    # Recursively auto-collect all `.py` files (excluding tests).
    # Note that `glob` won't recurse when subfolders have `BUILD` files. To
    # have a single top-level rule with additional `py_test` BUILD rules in
    # subfolders, see go/oss-kit#single-rule-pattern.
    srcs = glob_py_srcs(),
    data = [
        "src/ai_edge_eval/config/runners.yaml",
        "src/ai_edge_eval/config/tasks.yaml",
    ],
    # Project dependencies (matching the `pip install` deps defined in
    # `pyproject.toml`).
    deps = [
        "//third_party/odml/litert_lm/python/litert_lm",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/click",
        "//third_party/py/fastapi",
        "//third_party/py/httpx",
        "//third_party/py/huggingface_hub",
        "//third_party/py/lm_eval",
        "//third_party/py/pydantic:pydantic_v2",
        "//third_party/py/requests",
        "//third_party/py/torchvision",
        "//third_party/py/tqdm",
        "//third_party/py/uvicorn",
        "//third_party/py/yaml",
    ],
)

copybara_config_test(
    name = "copybara_test",
    config = "copy.bara.sky",
    deps = [
        "//third_party/py/etils:copybara_utils",
    ],
)

py_test(
    name = "litert_lm_server_test",
    srcs = ["src/ai_edge_eval/tests/unit/runners/litert_lm_server_test.py"],
    deps = [
        ":ai_edge_eval",
        "//third_party/py/fastapi",
    ],
)

py_test(
    name = "litert_lm_test",
    srcs = ["src/ai_edge_eval/tests/unit/runners/litert_lm_test.py"],
    deps = [":ai_edge_eval"],
)

py_test(
    name = "base_test",
    srcs = ["src/ai_edge_eval/tests/unit/runners/base_test.py"],
    deps = [":ai_edge_eval"],
)

py_test(
    name = "frameworks_base_test",
    srcs = ["src/ai_edge_eval/tests/unit/frameworks/base_test.py"],
    main = "src/ai_edge_eval/tests/unit/frameworks/base_test.py",
    deps = [":ai_edge_eval"],
)

py_test(
    name = "lm_eval_test",
    srcs = ["src/ai_edge_eval/tests/unit/frameworks/lm_eval_test.py"],
    deps = [":ai_edge_eval"],
)

py_test(
    name = "pipeline_test",
    srcs = ["src/ai_edge_eval/tests/unit/pipeline_test.py"],
    deps = [
        ":ai_edge_eval",
        "//third_party/py/yaml",
    ],
)

py_test(
    name = "main_test",
    srcs = ["src/ai_edge_eval/tests/unit/main_test.py"],
    deps = [
        ":ai_edge_eval",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/click",
    ],
)

py_binary(
    name = "cli",
    srcs = ["src/ai_edge_eval/cli/main.py"],
    main = "src/ai_edge_eval/cli/main.py",
    deps = [
        ":ai_edge_eval",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/click",
    ],
)

py_test(
    name = "local_chat_score_model_test",
    srcs = ["src/ai_edge_eval/tests/unit/frameworks/local_chat_score_model_test.py"],
    deps = [":ai_edge_eval"],
)

py_test(
    name = "generation_config_test",
    srcs = ["src/ai_edge_eval/tests/unit/generation_config_test.py"],
    deps = [
        ":ai_edge_eval",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/pydantic:pydantic_v2",
    ],
)

py_test(
    name = "custom_task_test",
    srcs = ["src/ai_edge_eval/tests/unit/custom_tasks/custom_task_test.py"],
    tags = ["no_py_strict_deps"],
    deps = [
        ":ai_edge_eval",
        "//third_party/py/absl/testing:absltest",
    ],
)

py_test(
    name = "registry_test",
    srcs = ["src/ai_edge_eval/tests/unit/custom_tasks/registry_test.py"],
    tags = ["no_py_strict_deps"],
    deps = [
        ":ai_edge_eval",
        "//third_party/py/absl/testing:absltest",
    ],
)

py_test(
    name = "loaders_test",
    srcs = ["src/ai_edge_eval/tests/unit/custom_tasks/loaders_test.py"],
    tags = ["no_py_strict_deps"],
    deps = [
        ":ai_edge_eval",
        "//third_party/py/absl/testing:absltest",
    ],
)

py_test(
    name = "custom_framework_test",
    srcs = ["src/ai_edge_eval/tests/unit/frameworks/custom_framework_test.py"],
    tags = ["no_py_strict_deps"],
    deps = [
        ":ai_edge_eval",
        "//third_party/py/absl/testing:absltest",
    ],
)

py_test(
    name = "introspection_test",
    srcs = ["src/ai_edge_eval/tests/unit/utils/introspection_test.py"],
    deps = [
        ":ai_edge_eval",
        "//third_party/py/pydantic:pydantic_v2",
    ],
)

py_test(
    name = "cli_list_args_test",
    srcs = ["src/ai_edge_eval/tests/unit/cli_list_args_test.py"],
    deps = [
        ":ai_edge_eval",
        "//third_party/py/click",
    ],
)
