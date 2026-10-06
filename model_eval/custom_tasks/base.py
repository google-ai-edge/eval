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
from typing import Any, Callable, Generic, Iterator, NotRequired, TypeVar, TypedDict

from model_eval import config
from model_eval.api import constants as api_constants

# Type alias for the request payloads of a dataset row: a list of JSON-object
# dictionaries POSTed in order to the target endpoint.
Requests = list[dict[str, Any]]

# Type alias for task-specific key-value arguments forwarded from `--eval-args`
# (merged on top of the runner's `server_args`, e.g. `model_name`).
TaskArgs = dict[str, Any]

# Type alias for the prediction type and ground truth type. They are not
# necessarily the same.
PredictionType = TypeVar("PredictionType")
GroundTruthType = TypeVar("GroundTruthType")


class DatasetRow(TypedDict, Generic[GroundTruthType]):
  """A dataset row with input requests, ground truth, and optional metadata."""

  requests: Requests
  ground_truth: GroundTruthType
  metadata: NotRequired[dict[str, Any]]


def chat_request(
    messages: list[dict[str, Any]],
    generation_config: config.GenerationConfig | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
  """Builds an OpenAI-compatible chat completions request body.

  Args:
    messages: OpenAI-style chat messages (`[{"role": ..., "content": ...}]`).
    generation_config: Optional generation parameters (temperature, max tokens,
      stop sequences). Defaults to `GenerationConfig()`.
    model_name: Optional model identifier for the `"model"` request field.
      Defaults to `DEFAULT_MODEL_NAME` when None or empty.

  Returns:
    A JSON-serializable request payload for `v1/chat/completions`.
  """
  cfg = generation_config or config.GenerationConfig()
  return {
      "model": model_name or api_constants.DEFAULT_MODEL_NAME,
      "messages": messages,
      "temperature": cfg.temperature,
      "max_tokens": cfg.max_new_tokens,
      "stop": cfg.stop_sequences or None,
  }


def chat_response_text(response: dict[str, Any] | None) -> str | None:
  """Extracts the assistant reply text from a chat completions response.

  Args:
    response: Decoded JSON response from `v1/chat/completions`, or None if the
      request failed.

  Returns:
    The first choice's message content, or None if `response` is None or does
    not contain `choices[0].message.content`.
  """
  if response is None:
    return None
  try:
    return response["choices"][0]["message"]["content"]
  except (KeyError, IndexError, TypeError):
    return None


@dataclasses.dataclass
class CustomTask(Generic[PredictionType, GroundTruthType]):
  """Definition of a custom evaluation task.

  Attributes:
    name: Unique string identifier for the custom task.
    endpoint: Server path (joined to the runner's `server_url`) that each row's
      `requests` are POSTed to, e.g. `"v1/chat/completions"` or
      `"v1/embeddings"`. Leading and trailing slashes are stripped.
    dataset: Local file path (.jsonl/.csv) or a callable `dataset(task_args)`
      returning an iterator of DatasetRow representing the dataset rows.
      `task_args` holds the runner's `server_args` (e.g. `model_name` from
      `--runner-args` for `http-server`) merged with the task-specific
      `--eval-args` entries; `--eval-args` win on key collisions.
    metric_fn: A function for computing evaluation metrics. Takes three parallel
      iterators of equal length: the generated predictions (each a list of
      per-request responses for that row), the expected ground truths, and the
      corresponding full DatasetRow objects, in matching order. Returns a
      dictionary mapping metric names to their computed values.
  """

  name: str
  endpoint: str
  dataset: str | Callable[[TaskArgs], Iterator[DatasetRow[GroundTruthType]]]
  metric_fn: Callable[
      [
          Iterator[PredictionType],
          Iterator[GroundTruthType],
          Iterator[DatasetRow[GroundTruthType]],
      ],
      dict[str, Any],
  ]

  def __post_init__(self) -> None:
    path = self.endpoint.strip("/") if isinstance(self.endpoint, str) else ""
    if not path:
      raise ValueError(
          "CustomTask.endpoint must be a non-empty path; got"
          f" {self.endpoint!r}."
      )
    self.endpoint = path
