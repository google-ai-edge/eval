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

"""Example Python plugins script registering reusable loaders and metrics for YAML catalogs."""

from model_eval import custom_tasks


@custom_tasks.register_loader("example_dummy_loader")
def load_dummy_dataset(
    spec: custom_tasks.TaskSpec,
) -> list[custom_tasks.DatasetRow]:
  """Produces dummy dataset entries based on task options."""
  count = spec.options.get("count", 2)
  prefix = spec.options.get("prefix", "Q:")
  rows = []
  for i in range(1, count + 1):
    rows.append(
        custom_tasks.DatasetRow(
            messages=[{"role": "user", "content": f"{prefix} {i} + {i}"}],
            ground_truth=str(i + i),
        )
    )
  return rows


@custom_tasks.register_metric("example_dummy_metric")
def compute_dummy_metric(
    preds: list[str],
    gts: list[str],
    rows: list[custom_tasks.DatasetRow],
    *,
    normalizer,
) -> dict[str, float]:
  """Computes dummy exact match accuracy."""
  p = [normalizer(text) for text in preds]
  g = [normalizer(text) for text in gts]
  if not p:
    return {"accuracy": 0.0}
  matches = sum(pi.strip() == gi for pi, gi in zip(p, g))
  return {"accuracy": matches / len(p)}
