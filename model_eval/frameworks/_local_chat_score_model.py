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

"""LiteRT LM chat score model for lm-eval."""

from typing import Any

from model_eval.api import constants as api_constants
from model_eval.frameworks import utils
from lm_eval.api import registry  # pylint: disable=g-importing-member
from lm_eval.models import openai_completions  # pylint: disable=g-importing-member
import requests as requests_lib
import tqdm


@registry.register_model(api_constants.LOCAL_CHAT_SCORE_MODEL_NAME)
class LocalChatScoreModel(openai_completions.LocalChatCompletion):
  """Custom scoring model that calls /v1/chat/score and delegates generation."""

  def __init__(
      self,
      base_url: str | None = None,
      model: str | None = None,
      **kwargs,
  ) -> None:
    if base_url is None:
      raise ValueError("Base URL must be provided.")
    self._root = base_url.rstrip("/")
    self._scoring_url = f"{self._root}/{api_constants.CHAT_SCORE_ENDPOINT}"
    super().__init__(
        base_url=f"{self._root}/{api_constants.CHAT_COMPLETIONS_ENDPOINT}",
        model=model,
        **kwargs,
    )

  def loglikelihood(
      self, requests: list[Any], disable_tqdm: bool = False
  ) -> list[tuple[float, bool]]:
    results = []
    if not requests:
      return results
    is_multimodal = len(requests[0].args) > 2
    if is_multimodal:
      raise ValueError("Multimodal scoring is not supported.")

    for req in tqdm.tqdm(requests, desc="Requesting API", disable=disable_tqdm):
      context, continuation = req.args

      # Build the message list.
      messages = utils.build_chat_score_messages(context, continuation)

      payload = {
          "model": self.model,
          "messages": messages,
      }

      response = requests_lib.post(self._scoring_url, json=payload)
      response.raise_for_status()
      data = response.json()

      # Extract the score and greedy flag from the response.
      score, is_greedy = utils.parse_chat_score_response(data)
      results.append((score, is_greedy))

    return results

  def loglikelihood_rolling(
      self, requests: list[Any], disable_tqdm: bool = False
  ) -> list[float]:
    raise NotImplementedError("loglikelihood_rolling is not implemented.")
