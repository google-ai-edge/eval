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

"""LiteRT LM runner implementation."""

import os
import threading
from typing import Any, Callable

from model_eval.api import constants as api_constants
from model_eval.runners import base
from model_eval.runners import registry
from model_eval.runners.litert_lm import _litert_lm_server
import requests
import uvicorn

import litert_lm

_DEFAULT_TIMEOUT_SECONDS = 300


def _resolve_model_path(path: str) -> str:  # pylint: disable=g-doc-args
  """Resolves a model path checking local filesystem first, then downloading from HuggingFace.

    Model path examples:
      - Absolute path: "/abs/path/to/model.litertlm"
      - Relative path: "models/model.litertlm"
      - HuggingFace identifier (repo_id / model_path_in_repo):
        "org/repo/path/to/model.litertlm"

  Args:
    path: Absolute or relative path, or HuggingFace identifier.

  Returns:
    The resolved local absolute path to the model.
  """
  if os.path.exists(path):
    return path

  parts = path.split("/")
  if len(parts) >= 3 and not path.startswith("/"):
    # Repo ID is the first two parts, i.e. "org/repo".
    repo_id = "/".join(parts[:2])
    # The model path within the repo can be anything.
    filename = "/".join(parts[2:])
    try:
      # Lazy-import to avoid making HuggingFace a hard dependency for users who
      # only run LiteRT-LM on pre-downloaded models.
      import huggingface_hub  # pylint: disable=g-import-not-at-top

      downloaded_path = huggingface_hub.hf_hub_download(
          repo_id=repo_id, filename=filename
      )
      if downloaded_path:
        return downloaded_path
    except Exception as e:
      raise RuntimeError(
          f"Failed to download '{filename}' from HuggingFace repository"
          f" '{repo_id}': {e}"
      ) from e
  return path


def _parse_backend(backend_str: str) -> litert_lm.Backend:
  """Parses a string backend to the litert_lm.Backend."""
  if isinstance(backend_str, litert_lm.Backend):
    return backend_str
  backend_upper = backend_str.upper()
  if backend_upper == "CPU":
    return litert_lm.Backend.CPU()
  elif backend_upper == "GPU":
    return litert_lm.Backend.GPU()
  elif backend_upper == "NPU":
    return litert_lm.Backend.NPU()
  else:
    valid_backends = ["CPU", "GPU", "NPU"]
    raise ValueError(
        f"Unsupported backend: '{backend_str}'. Must be one of {valid_backends}"
    )


def _parse_activation_data_type(
    activation_data_type: str,
) -> litert_lm.ActivationDataType:
  """Parses a string activation data type to the litert_lm.ActivationDataType."""
  valid_types = ["fp32", "fp16"]
  if activation_data_type not in valid_types:
    raise ValueError(
        f"Unsupported activation data type: '{activation_data_type}'. "
        f"Must be one of {valid_types}."
    )
  parsed = litert_lm.ActivationDataType.from_str(activation_data_type)
  if parsed is None:
    raise ValueError(
        f"Failed to parse activation data type: '{activation_data_type}'"
    )
  return parsed


def _clamp_log_severity(severity: int) -> litert_lm.LogSeverity:
  """Clamps a given log severity level to the supported litert_lm.LogSeverity."""
  if severity <= 0:
    return litert_lm.LogSeverity.VERBOSE
  if severity >= 1000:
    return litert_lm.LogSeverity.SILENT
  if severity > 5:
    return litert_lm.LogSeverity.FATAL
  return litert_lm.LogSeverity(severity)


@registry.register_runner(base.RunnerType.LITERT_LM)
class LiteRtLmRunner(base.AbstractRunner):
  """Runner using LiteRT LM backend.

  Exposes two primary HTTP endpoints:
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

  class Config(base.RunnerConfig):
    """Configuration for LiteRtLmRunner."""

    # Path to the LiteRT LM model file.
    model_path: str
    # Backend to use for the LiteRT LM model.
    backend: str = "cpu"
    # Optional vision backend.
    vision_backend: str | None = None
    # Optional audio backend.
    audio_backend: str | None = None
    # Maximum number of tokens for KV cache.
    max_num_tokens: int = 4096
    # Whether to enable speculative decoding.
    enable_speculative_decoding: bool | None = None
    # Optional activation data type to use for the model (e.g. 'fp32', 'fp16').
    activation_data_type: str | None = None
    # Host for the runner's server.
    host: str = "127.0.0.1"
    # Port for the runner's server.
    port: int = 8080
    # Minimum logging severity level:
    #   <= 0:    LogSeverity.VERBOSE
    #   1:       LogSeverity.DEBUG
    #   2:       LogSeverity.INFO
    #   3:       LogSeverity.WARNING
    #   4:       LogSeverity.ERROR
    #   5:       LogSeverity.FATAL
    #   >= 1000: LogSeverity.SILENT
    min_log_severity: int = 1000
    # Whether to skip the slow run_decode step for greedy verification.
    always_return_not_greedy: bool = True
    # Whether to enable text and multimodal scoring.
    enable_scoring: bool = True
    # Whether to enable thinking/reasoning generation.
    thinking: bool | None = None
    # Budget for reasoning tokens. 0 disables thinking. -1 enables unlimited.
    thinking_budget: int | None = None

    @classmethod
    def from_unified_args(
        cls,
        model_path: str | None,
        device: str | None,
        runner_args: dict[str, Any],
    ) -> "LiteRtLmRunner.Config":
      # Avoid conflicts between unified CLI flags and runner_args.
      if "model_path" in runner_args and model_path:
        raise ValueError(
            "--model-path conflicts with 'model_path' in --runner-args. Use one"
            " or the other."
        )
      if "backend" in runner_args and device:
        raise ValueError(
            "--device conflicts with 'backend' in --runner-args. Use one or the"
            " other."
        )
      if model_path:
        runner_args["model_path"] = model_path
      if device:
        runner_args["backend"] = device
      runner_args["runner_type"] = "litert-lm"
      return cls(**runner_args)

  config_type: type[Config] = Config
  _config: Config

  def __init__(self, config: Config, *args: Any, **kwargs: Any) -> None:  # pylint: disable=unused-argument
    super().__init__()
    self._config = config
    self._server_thread: threading.Thread | None = None
    self._server: uvicorn.Server | None = None
    self._engine: litert_lm.Engine | None = None

  def start(self) -> None:
    litert_lm.set_min_log_severity(
        _clamp_log_severity(self._config.min_log_severity)
    )
    engine_kwargs = {}
    if self._config.vision_backend:
      engine_kwargs["vision_backend"] = _parse_backend(
          self._config.vision_backend
      )
    if self._config.audio_backend:
      engine_kwargs["audio_backend"] = _parse_backend(
          self._config.audio_backend
      )
    if self._config.enable_speculative_decoding is not None:
      engine_kwargs["enable_speculative_decoding"] = (
          self._config.enable_speculative_decoding
      )
    if self._config.activation_data_type is not None:
      engine_kwargs["activation_data_type"] = _parse_activation_data_type(
          self._config.activation_data_type
      )

    # Resolve the model path (download from HuggingFace if necessary).
    path = _resolve_model_path(self._config.model_path)
    self._engine = litert_lm.Engine(
        path,
        backend=_parse_backend(self._config.backend),
        max_num_tokens=self._config.max_num_tokens,
        **engine_kwargs,
    )
    # The litert_lm C++ backend doesn't take host/port directly but we're
    # wrapping it via FastAPI.
    app = _litert_lm_server.build_app(self._engine, self._config)
    uv_config = uvicorn.Config(
        app,
        host=self._config.host,
        port=self._config.port,
        log_level="warning",
        access_log=False,
    )
    self._server = uvicorn.Server(uv_config)

    self._server_thread = threading.Thread(target=self._server.run, daemon=True)
    self._server_thread.start()
    _litert_lm_server.wait_for_server(
        self.server_url, timeout=_DEFAULT_TIMEOUT_SECONDS
    )

    self._validate_completions()
    if self._config.enable_scoring:
      self._validate_scoring()

  def stop(self) -> None:
    if self._server is not None:
      self._server.should_exit = True
    if self._server_thread is not None:
      self._server_thread.join()
    self._server = None
    self._server_thread = None
    self._engine = None

  def _validate_completions(self) -> None:
    """Verifies the /v1/chat/completions endpoint."""
    url = f"{self.server_url}/{api_constants.CHAT_COMPLETIONS_ENDPOINT}"
    payload = {
        "model": api_constants.DEFAULT_MODEL_NAME,
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
        "model": api_constants.DEFAULT_MODEL_NAME,
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

  @property
  def server_url(self) -> str:
    return f"http://{self._config.host}:{self._config.port}"

  @property
  def returns_greedy(self) -> bool:
    return not self._config.always_return_not_greedy
