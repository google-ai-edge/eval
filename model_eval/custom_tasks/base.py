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

"""Base definitions for the custom task layer."""

import dataclasses
from typing import Any, Callable, Generic, Iterator, NotRequired, TypedDict, TypeVar

from model_eval import config

# Type alias for OpenAI messages format (list of dictionaries with 'role' and
# 'content').
OpenAIMessages = list[dict]

# Type alias for the prediction type and ground truth type. They are not
# necessarily the same.
PredictionType = TypeVar("PredictionType")
GroundTruthType = TypeVar("GroundTruthType")


class DatasetRow(TypedDict, Generic[GroundTruthType]):
  """A dataset row separating input context and expected output with optional metadata."""

  messages: OpenAIMessages
  ground_truth: GroundTruthType
  metadata: NotRequired[dict[str, Any]]


@dataclasses.dataclass
class CustomTask(Generic[PredictionType, GroundTruthType]):
  """Definition of a generation-only custom evaluation task.

  Attributes:
    name: Unique string identifier for the custom task.
    dataset: Local file path (.jsonl/.csv) or a callable returning an iterator
      of DatasetRow representing the dataset rows.
    metric_fn: A function for computing evaluation metrics. Takes three parallel
      iterators of equal length: the generated predictions, the expected ground
      truths, and the corresponding full DatasetRow objects, in matching
      order. Returns a dictionary mapping metric names to their computed values.
    generation_config: Configuration for per-request generation parameters.
  """

  name: str
  dataset: str | Callable[[], Iterator[DatasetRow[GroundTruthType]]]  # pytype: disable=invalid-annotation
  metric_fn: Callable[
      [Iterator[PredictionType], Iterator[GroundTruthType], Iterator[DatasetRow[GroundTruthType]]], dict[str, Any]  # pytype: disable=invalid-annotation
  ]
  generation_config: config.GenerationConfig = dataclasses.field(
      default_factory=config.GenerationConfig
  )
