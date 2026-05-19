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

"""Dataset loading and slicing utility."""

import csv
import json
import pathlib
from typing import Callable, Iterator

from model_eval.custom_tasks import base

# Default column name for DatasetRow in CSV files.
CSV_DATA_COLUMN = "data"


def parse_samples(expr: str, n: int) -> list[int]:
  """Parse slice expressions like '0-99', '0,2,4', '::2', '10:50:2'.

  Args:
    expr: The slicing expression string representing ranges, slices, or indices.
    n: The upper bound limit of indices available in the dataset to validate
      against.

  Returns:
    A list of valid row indices extracted from the string expression.
  """
  indices = []
  # Handle range format (e.g., "10-20") inclusive of both ends.
  if "-" in expr:
    start_str, end_str = expr.split("-")
    start = max(0, int(start_str))
    end = min(n, int(end_str) + 1)
    indices = list(range(start, end))
  # Handle standard slice notation format (e.g., "10:50:2").
  elif ":" in expr:
    parts = [int(p) if p else None for p in expr.split(":")]
    start = parts[0] or 0
    stop = parts[1] or n
    step = parts[2] if len(parts) > 2 and parts[2] else 1
    start = max(0, start)
    stop = min(n, stop)
    indices = list(range(start, stop, step))
  # Handle comma-separated specific indices (e.g., "0,2,4").
  else:
    for part in expr.split(","):
      part = part.strip()
      if not part:
        continue
      val = int(part)
      if 0 <= val < n:
        indices.append(val)
  return indices


def load_dataset(
    source: str | Callable[[], Iterator[base.DatasetRow]],
) -> Iterator[base.DatasetRow]:
  """Load DatasetRow from JSONL, CSV, or a generator Callable.

  Args:
    source: Local string file path to a dataset or a generator yielding
      DatasetRow.

  Yields:
    Parsed DatasetRow representing individual conversational turn sequences.
  """
  # Forward directly if the source is already a callable generator.
  if callable(source):
    yield from source()
  else:
    path = pathlib.Path(source)
    # For JSONL files, each line is expected to be a JSON-encoded DatasetRow
    # object.
    if path.suffix == ".jsonl":
      for line in path.read_text().splitlines():
        if line.strip():
          yield json.loads(line)
    # For CSV files, rows must contain a "data" column with JSON-encoded
    # DatasetRow object.
    elif path.suffix == ".csv":
      with path.open() as f:
        for row in csv.DictReader(f):
          yield json.loads(row[CSV_DATA_COLUMN])
    else:
      raise ValueError(
          f"Unsupported file type: {path.suffix}. Use .jsonl or .csv"
      )
