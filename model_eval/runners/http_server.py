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
        runner_args: Must contain `server_url` (the server's base URL). Optional
          `request_timeout_sec` sets the per-request timeout. Any other key is
          kept verbatim in `server_args`.

      Returns:
        The runner config.

      Raises:
        ValueError: If `server_url` is missing or invalid, or
          `request_timeout_sec` is not a positive number.
      """
      del model_path, device  # The model is served remotely.
      server_url = runner_args.get("server_url")
      if not server_url:
        raise ValueError(
            "server_url is required in runner_args for http-server runner."
        )
      # Keys naming a declared Config field belong to the runner; everything
      # else is passed through untouched as `server_args`.
      server_args = {
          key: value
          for key, value in runner_args.items()
          if key not in cls.model_fields
      }
      return cls(
          runner_type=base.RunnerType.HTTP_SERVER,
          server_url=_validate_server_url(str(server_url)),
          request_timeout_sec=runner_args.get("request_timeout_sec"),
          server_args=server_args,
      )

  config_type: type[Config] = Config
  _config: Config

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
