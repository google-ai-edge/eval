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

"""LiteRT LM model client for lighteval."""

from typing import Any, List

from model_eval.api import constants as api_constants
from model_eval.frameworks import utils
import httpx
from lighteval.models import model_output  # type: ignore
from lighteval.models.endpoints import litellm_model  # type: ignore


class ChatScoreLightevalModel(litellm_model.LiteLLMClient):
  """LiteLLMClient subclass that adds scoring via the API endpoint.

  To be compatible with the Lighteval pipeline, a custom model adapter must
  implement specific programmatic interfaces evaluated across different task
  types (such as `generate_responses` for generative tasks and `loglikelihood`
  for multiple-choice tasks).

  Subclassing `LiteLLMClient` is sufficient because it already implements the
  required base model interfaces, naturally handling generative tasks by
  falling back to the standard OpenAI completions API. We only need to
  specifically override `loglikelihood` to perform multiple-choice scoring
  using our unique `/chat/score` endpoint.
  """

  def __init__(self, config: Any):
    super().__init__(config)
    # config.base_url contains /v1 (e.g. http://127.0.0.1:8080/v1)
    # We strip it here because CHAT_SCORE_ENDPOINT is already "v1/chat/score"
    self._server_url = config.base_url.rstrip("/")
    if self._server_url.endswith("/v1"):
      self._server_url = self._server_url[:-3]
    self._score_url = f"{self._server_url}/{api_constants.CHAT_SCORE_ENDPOINT}"
    self._model_name = config.model_name

  def loglikelihood(
      self, docs: list[Any], override_bs: Any = None
  ) -> List[model_output.ModelResponse]:
    responses = []
    with httpx.Client(timeout=120.0) as client:
      for doc in docs:
        context = doc.query
        scores = []
        is_greedys = []
        for continuation in doc.choices:
          # Build the message list.
          messages = utils.build_chat_score_messages(context, continuation)

          payload = {
              "model": self._model_name,
              "messages": messages,
          }

          resp = client.post(self._score_url, json=payload)
          resp.raise_for_status()
          data = resp.json()

          # Extract the score and greedy flag from the response.
          score, is_greedy = utils.parse_chat_score_response(data)
          scores.append(score)
          is_greedys.append(is_greedy)

        responses.append(
            model_output.ModelResponse(
                logprobs=scores,
                argmax_logits_eq_gold=is_greedys,
            )
        )
    return responses

  def loglikelihood_rolling(
      self, docs: Any, override_bs: Any = None
  ) -> List[Any]:
    raise NotImplementedError("loglikelihood_rolling is not implemented.")
