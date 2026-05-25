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

"""Base runner abstractions."""

import abc
import dataclasses
import enum
from typing import Any

from model_eval.api import constants as api_constants
from model_eval.utils import introspection
import pydantic
import requests

_DEFAULT_TIMEOUT_SECONDS = 300


class RunnerType(enum.StrEnum):
  """Supported runner types."""

  LITERT_LM = "litert-lm"


@dataclasses.dataclass
class RunnerCapabilities:
  """Flags detailing the supported functionality of a runner."""

  # We by default support text scoring and generation.
  text_scoring: bool = True
  text_generation: bool = True
  # Multimodal capabilities are optional.
  multimodal_generation: bool = False
  multimodal_scoring: bool = False


class RunnerConfig(pydantic.BaseModel):
  """Configuration base class for runner definitions."""

  runner_type: str

  @classmethod
  @abc.abstractmethod
  def from_unified_args(
      cls,
      model_path: str | None,
      device: str | None,
      runner_args: dict[str, Any],
  ) -> "RunnerConfig":
    """Translate unified flags into this runner's own config field names."""
    ...


class AbstractRunner(abc.ABC):
  """Base abstract interface for server-based runner implementations.

  Server implementations are expected to expose two primary endpoints:
  1. `/v1/chat/completions`: Used for standard chat generation, adhering to the
     standard OpenAI API schema.
  2. `/v1/chat/score`: Custom endpoint for scoring a chat continuation. The last
     message in the request must have `role="assistant"` (representing the
     continuation string to be scored), while all preceding messages form the
     context.
     It returns results structured according to the OpenAI `/v1/completions`
     response schema (using a `choices` list and a `logprobs` object), but
     simplifies client-side evaluation by returning aggregate scoring metadata:
     {
       "choices": [
         {
           "score": float,          # Server-computed loglikelihood score for
           the continuation.
           "logprobs": {
             "is_greedy": bool,     # True if the continuation matches greedy
             decoding.
             ...
           }
         }
       ]
     }
  """

  config: type[RunnerConfig] = RunnerConfig

  @abc.abstractmethod
  def start(self) -> None:
    """Start the OpenAI-compatible HTTP server. Blocks until ready."""

  @abc.abstractmethod
  def stop(self) -> None:
    """Stop the server and release all resources."""

  @property
  @abc.abstractmethod
  def server_url(self) -> str:
    """Base URL, e.g. 'http://127.0.0.1:8080'."""

  @property
  @abc.abstractmethod
  def model_name(self) -> str:
    """Model identifier sent to frameworks as the model= parameter."""

  @property
  @abc.abstractmethod
  def capabilities(self) -> RunnerCapabilities:
    """Returns the capabilities profile of the runner."""
    ...

  @classmethod
  def describe_runner_args(cls) -> list[dict[str, Any]]:
    """Returns descriptions of accepted runner arguments."""
    return introspection.get_fields(cls.config)

  def __enter__(self) -> "AbstractRunner":
    # Use reference counting to handle nested context manager entries.
    if getattr(self, "_ref_count", 0) == 0:
      self.start()
      self._ref_count = 0
    self._ref_count += 1
    return self

  def __exit__(self, *_) -> None:
    if hasattr(self, "_ref_count"):
      self._ref_count -= 1
      if self._ref_count == 0:
        self.stop()

  def _validate_completions(self) -> None:
    """Verifies the /v1/chat/completions endpoint."""
    url = f"{self.server_url}/{api_constants.CHAT_COMPLETIONS_ENDPOINT}"
    payload = {
        "model": self.model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    try:
      response = requests.post(
          url, json=payload, timeout=_DEFAULT_TIMEOUT_SECONDS
      )
      response.raise_for_status()
      if "choices" not in response.json():
        raise ValueError("Response missing 'choices' field.")
    except Exception as e:
      raise RuntimeError(
          f"Runner failed generation validation at {url}: {e}"
      ) from e

  def _validate_scoring(self) -> None:
    """Verifies the /v1/chat/score endpoint."""
    url = f"{self.server_url}/{api_constants.CHAT_SCORE_ENDPOINT}"
    payload = {
        "model": self.model_name,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    }
    try:
      response = requests.post(
          url, json=payload, timeout=_DEFAULT_TIMEOUT_SECONDS
      )
      response.raise_for_status()
      data = response.json()
      choice = data["choices"][0]
      if "score" not in choice or "logprobs" not in choice:
        raise ValueError("Response missing 'score' or 'logprobs' fields.")
    except Exception as e:
      raise RuntimeError(
          f"Runner failed scoring validation at {url}: {e}"
      ) from e
