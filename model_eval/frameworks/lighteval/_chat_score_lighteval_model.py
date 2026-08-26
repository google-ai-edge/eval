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

import base64
import io
from typing import Any, List, cast

from model_eval.api import constants as api_constants
from model_eval.frameworks import utils
import httpx
from lighteval.models import model_output  # type: ignore
from lighteval.models.endpoints import litellm_model  # type: ignore
from lighteval.tasks import prompt_manager  # type: ignore


def _encode_image_to_data_uri(img: Any) -> str:
  """Converts a PIL Image to a base64-encoded JPEG data URI.

  Args:
    img: A PIL Image instance to encode.

  Returns:
    A string formatted as 'data:image/jpeg;base64,...'.
  """
  buf = io.BytesIO()
  # Convert images with alpha channels (e.g., RGBA, LA, P) to RGB to avoid
  # OSError when saving as JPEG, which does not support transparency.
  img_rgb = img.convert("RGB") if getattr(img, "mode", "RGB") != "RGB" else img
  img_rgb.save(buf, format="JPEG")
  b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
  return f"data:image/jpeg;base64,{b64_str}"


def _inject_images_into_content(
    content: Any, images: list[Any]
) -> list[dict[str, Any]]:
  """Converts string or list content into structured multimodal blocks.

  Args:
    content: The original message content, either a string or a list of content
      dictionaries.
    images: A list of PIL Image objects to encode and append.

  Returns:
    A list of content dictionaries containing text and image_url items.
  """
  new_content: list[dict[str, Any]]
  if isinstance(content, str):
    new_content = [{"type": "text", "text": content}]
  elif isinstance(content, list):
    new_content = list(content)
  else:
    raise ValueError(f"Unsupported message content type: {type(content)}")

  # Append images at the end of the text. This text-first, images-last order
  # matches Lighteval's own reference implementation: http://shortn/_OjtqEiHT7O.
  new_content.extend([
      {
          "type": "image_url",
          "image_url": {"url": _encode_image_to_data_uri(img)},
      }
      for img in images
  ])
  return new_content


class MultimodalPromptManager(prompt_manager.PromptManager):
  """Multimodal-aware prompt manager for API-based model clients."""

  def prepare_prompt_api(self, doc: Any) -> list[dict[str, Any]]:
    """Prepares a list of message dictionaries for API-based chat models.

    Builds the standard text chat messages using the upstream PromptManager,
    then injects base64-encoded JPEG image data URIs into the user messages
    for both few-shot samples and the main evaluation query.

    Args:
      doc: A lighteval Doc instance containing the query, instruction, few-shot
        samples, and optional images.

    Returns:
      A list of message dictionaries formatted for OpenAI/LiteLLM APIs.
    """
    # Let the base PromptManager build the standard message structure.
    base_messages = super().prepare_prompt_api(doc)

    # Check whether there are any images in either the main doc or few-shots.
    fewshots = getattr(doc, "fewshot_samples", None) or []
    has_images = getattr(doc, "images", None) or any(
        getattr(sample, "images", None) for sample in fewshots
    )

    # Create shallow copies of messages to prevent mutating cached structures.
    messages: list[dict[str, Any]] = [
        cast(dict[str, Any], msg.copy()) for msg in base_messages
    ]

    if not has_images:
      return messages

    # Calculate the message index offset when a system prompt is present.
    offset = 1 if self.system_prompt is not None else 0

    # Map images for few-shot examples (user messages occur every two messages).
    for i, sample in enumerate(fewshots):
      sample_images = getattr(sample, "images", None)
      if sample_images:
        msg_idx = offset + (2 * i)
        if msg_idx < len(messages) and messages[msg_idx].get("role") == "user":
          messages[msg_idx]["content"] = _inject_images_into_content(
              messages[msg_idx].get("content"), sample_images
          )

    # Map images for the main evaluation query (always the last message).
    main_images = getattr(doc, "images", None)
    if main_images and messages:
      last_idx = len(messages) - 1
      if messages[last_idx].get("role") == "user":
        messages[last_idx]["content"] = _inject_images_into_content(
            messages[last_idx].get("content"), main_images
        )

    return messages


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

    self.prompt_manager = MultimodalPromptManager(
        use_chat_template=True,
        tokenizer=self.tokenizer,
        system_prompt=config.system_prompt,
    )

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
