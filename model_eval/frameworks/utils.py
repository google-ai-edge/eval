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

"""Shared utility functions for evaluation frameworks."""

import json
from typing import Any
from absl import logging


def build_chat_score_messages(
    context: Any, continuation: str
) -> list[dict[str, Any]]:
  """Builds a message list for chat scoring endpoints from context and continuation.

  Args:
    context: The prompt context. Supported formats: - A string (treated as a
      single user turn). - A structured object with a `prompt` attribute
      containing a JSON-serialized list of chat messages (e.g., JsonChatStr from
      lm-eval). - An empty or falsy value (e.g., None, ""), which results in no
      context.
    continuation: The assistant continuation string to be scored.

  Returns:
    A list of chat messages in standard OpenAI format, ending with the
    continuation
    assigned to the "assistant" role.
  """
  if hasattr(context, "prompt"):
    # Unpack conversation history if it's a JsonChatStr from lm-eval's
    # apply_chat_template.
    messages = json.loads(context.prompt)
  elif not context:
    logging.warning("Empty context provided to build_chat_score_messages.")
    messages = []
  else:
    # For plain text contexts, treat as a single user turn.
    messages = [{"role": "user", "content": context}]

  # We always put the continuation in the assistant role.
  messages.append({"role": "assistant", "content": continuation})

  return messages


def parse_chat_score_response(response: dict[str, Any]) -> tuple[float, bool]:
  """Extracts the loglikelihood score and greedy flag from the /chat/score API response."""

  # Check for the existence of 'choices' and ensure the list is not empty.
  if (
      "choices" not in response
      or not isinstance(response["choices"], list)
      or len(response["choices"]) == 0
  ):
    raise ValueError(
        "Invalid API response: 'choices' list is missing or empty. Response:"
        f" {response}"
    )

  choice = response["choices"][0]

  # Verify the required keys exist in the first choice.
  if "score" not in choice:
    raise ValueError(
        "Invalid API response: 'score' key is missing in the first choice."
        f" Choice: {choice}"
    )

  if (
      "logprobs" not in choice
      or not isinstance(choice["logprobs"], dict)
      or "is_greedy" not in choice["logprobs"]
  ):
    raise ValueError(
        "Invalid API response: 'logprobs' dict or 'is_greedy' flag missing."
        f" Choice: {choice}"
    )

  score = float(choice["score"])
  is_greedy = bool(choice["logprobs"]["is_greedy"])

  return score, is_greedy
