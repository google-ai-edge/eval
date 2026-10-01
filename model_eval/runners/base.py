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
import enum
from typing import Any

from model_eval.utils import introspection
import pydantic


class RunnerType(enum.StrEnum):
  """Supported runner types."""

  LITERT_LM = "litert-lm"


class RunnerConfig(pydantic.BaseModel):
  """Configuration base class for runner definitions."""

  runner_type: str
  # Per-request timeout in seconds; None means each framework's own default.
  request_timeout_sec: float | None = None

  @pydantic.field_validator("request_timeout_sec", mode="before")
  @classmethod
  def _validate_request_timeout_sec(cls, timeout_sec: Any) -> float | None:
    """Returns `timeout_sec` as a positive float, or None if unset."""
    if timeout_sec is None:
      return None
    try:
      value = float(timeout_sec)
    except (TypeError, ValueError):
      value = 0.0
    if isinstance(timeout_sec, bool) or not value > 0:
      raise ValueError(
          f"request_timeout_sec must be a positive number; got {timeout_sec!r}."
      )
    return value

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
  """Base abstract interface for server-based runner implementations."""

  config: type[RunnerConfig] = RunnerConfig

  @abc.abstractmethod
  def start(self) -> None:
    """Start the HTTP server. Blocks until ready."""

  @abc.abstractmethod
  def stop(self) -> None:
    """Stop the server and release all resources."""

  @property
  @abc.abstractmethod
  def server_url(self) -> str:
    """Base URL, e.g. 'http://127.0.0.1:8080'."""

  @property
  @abc.abstractmethod
  def endpoints(self) -> tuple[str, ...]:
    """Server paths (without leading/trailing slashes) served by this runner."""

  @property
  @abc.abstractmethod
  def request_timeout_sec(self) -> float | None:
    """Per-request timeout in seconds; None means the framework's default."""

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
