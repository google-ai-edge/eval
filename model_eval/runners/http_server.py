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

"""Runner that targets an already-running HTTP model server."""

from typing import Any
from urllib import parse

from model_eval.runners import base
from model_eval.runners import registry
import pydantic


def _validate_server_url(server_url: str) -> str:
  """Returns `server_url` without a trailing slash.

  Args:
    server_url: Base URL of the server, e.g. "http://10.0.0.1:8080".

  Raises:
    ValueError: If `server_url` is not an http(s) URL.
  """
  parsed = parse.urlparse(server_url)
  if parsed.scheme not in ("http", "https") or not parsed.netloc:
    raise ValueError(
        f"server_url must be an http(s) URL, e.g. http://host:port; got"
        f" {server_url!r}."
    )
  return server_url.rstrip("/")


def _validate_endpoints(endpoints: Any) -> tuple[str, ...]:
  """Returns `endpoints` as a non-empty tuple of slash-stripped paths.

  Args:
    endpoints: A single endpoint path string or a sequence of endpoint paths,
      e.g. "v1/embeddings" or ["v1/chat/completions", "v1/chat/score"].

  Raises:
    ValueError: If `endpoints` is empty or contains an empty path.
  """
  if isinstance(endpoints, str):
    raw_items: list[Any] = [endpoints]
  elif isinstance(endpoints, (list, tuple)):
    raw_items = list(endpoints)
  else:
    raise ValueError(
        "endpoints must be a path string or a list of path strings; got"
        f" {endpoints!r}."
    )
  cleaned: list[str] = []
  for item in raw_items:
    if not isinstance(item, str):
      raise ValueError(
          f"Each endpoint must be a non-empty string; got {item!r}."
      )
    path = item.strip("/")
    if not path:
      raise ValueError(
          f"Each endpoint must be a non-empty path; got {item!r}."
      )
    cleaned.append(path)
  if not cleaned:
    raise ValueError("endpoints must contain at least one path.")
  return tuple(cleaned)


@registry.register_runner(base.RunnerType.HTTP_SERVER)
class HttpServerRunner(base.AbstractRunner):
  """Runner that talks to an existing HTTP model server.

  The server's lifecycle is managed outside the eval, so `start` and `stop`
  do nothing.
  """

  class Config(base.RunnerConfig):
    """Configuration for HttpServerRunner."""

    # Base URL of the server, e.g. "http://10.0.0.1:8080".
    server_url: str
    # Server paths (joined to `server_url`) that the server exposes, e.g.
    # ("v1/chat/completions", "v1/chat/score") or ("v1/embeddings",).
    endpoints: tuple[str, ...]

    @pydantic.field_validator("endpoints", mode="before")
    @classmethod
    def _normalize_endpoints(cls, endpoints: Any) -> tuple[str, ...]:
      return _validate_endpoints(endpoints)

    @classmethod
    def from_unified_args(
        cls,
        model_path: str | None,
        device: str | None,
        runner_args: dict[str, Any],
    ) -> "HttpServerRunner.Config":
      """Builds the config from `--runner-args`.

      Args:
        model_path: Unused; the model is served remotely.
        device: Unused; the model is served remotely.
        runner_args: Must contain `server_url` (the server's base URL) and
          `endpoints` (a path string or list of path strings exposed by the
          server). Optional `request_timeout_sec` sets the per-request timeout.

      Returns:
        The runner config.

      Raises:
        ValueError: If `server_url` or `endpoints` is missing or invalid, or
          `request_timeout_sec` is not a positive number.
      """
      del model_path, device  # The model is served remotely.
      server_url = runner_args.get("server_url")
      if not server_url:
        raise ValueError(
            "server_url is required in runner_args for http-server runner."
        )
      if "endpoints" not in runner_args or runner_args["endpoints"] is None:
        raise ValueError(
            "endpoints is required in runner_args for http-server runner."
        )
      return cls(
          runner_type=base.RunnerType.HTTP_SERVER,
          server_url=_validate_server_url(str(server_url)),
          endpoints=_validate_endpoints(runner_args["endpoints"]),
          request_timeout_sec=runner_args.get("request_timeout_sec"),
      )

  config: type[Config] = Config

  def __init__(self, config: Config, *args: Any, **kwargs: Any) -> None:  # pylint: disable=unused-argument
    super().__init__()
    self._config = config

  def start(self) -> None:
    """Does nothing, as the server is started externally."""

  def stop(self) -> None:
    """Does nothing, as the server is not managed by this runner."""

  @property
  def server_url(self) -> str:
    return self._config.server_url

  @property
  def endpoints(self) -> tuple[str, ...]:
    return self._config.endpoints

  @property
  def request_timeout_sec(self) -> float | None:
    return self._config.request_timeout_sec
