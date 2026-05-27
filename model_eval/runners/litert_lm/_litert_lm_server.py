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

"""FastAPI wrapper around LiteRT LM engine."""

import time
from typing import Any

from model_eval.api import constants as api_constants
from model_eval.runners import base
import fastapi
import pydantic
import requests

import litert_lm


class _ChatMessage(pydantic.BaseModel):
  """A single message in a chat request."""

  # Role of the message,  one of "user", "assistant", "system".
  role: str
  # str for text-only; list[dict] for multimodal content parts.
  content: str | list[dict[str, Any]]


class _ChatScoreRequest(pydantic.BaseModel):
  """A request to score chat completions."""

  # OpenAI spec - accepted but unused.
  model: str
  # Sequence of conversation messages ending with the continuation string to
  # score.
  messages: list[_ChatMessage]


class _ChatRequest(pydantic.BaseModel):
  """A request to generate chat completions."""

  # OpenAI spec - accepted but unused.
  model: str
  # Body of the request, including the user query and any preface messages.
  messages: list[_ChatMessage]
  # passed as SamplerConfig(temperature=...) to create_conversation()
  temperature: float = 0.0
  # Applied server-side after send_message() returns;
  # truncates at earliest matching stop sequence.
  stop: list[str] | str | None = None
  # The limit on tokens produced after the prompt. Lighteval treats `max_tokens`
  # and `max_completion_tokens` as interchangeable aliases.
  # Accepted on the wire so that pydantic doesn't reject lighteval/litellm
  # requests, but NOT enforced. The underlying litert_lm Conversation API
  # exposes no per-call token-budget hook.
  max_tokens: int | None = None
  max_completion_tokens: int | None = None


def _to_litert_lm_message(msg: _ChatMessage) -> dict[str, Any]:
  """Converts a chat message to a format accepted by the LiteRT LM wrapper."""
  if isinstance(msg.content, str):
    return {"role": msg.role, "content": msg.content}

  parts = []
  for part in msg.content:
    if part["type"] == "text":
      # CAVEAT: if there are <image> placeholders in the text, lm-eval's API
      # models will not split or interleave them with the image parts. The text
      # will be treated as a single part, which is different from their
      # HuggingFace models. This may lead to discrepancies in the results
      # between the two.
      parts.append({"type": "text", "text": part["text"]})
    elif part["type"] == "image_url":
      url = part["image_url"]["url"]
      if url.startswith("data:"):
        encoded = url.split(",", 1)[1]  # strip "data:image/...;base64,"
        parts.append({"type": "image", "blob": encoded})
      else:
        raise fastapi.HTTPException(
            status_code=400,
            detail=(
                "HTTP image URLs are not supported; send base64 data URLs"
                " instead"
            ),
        )
    elif part["type"] == "input_audio":
      parts.append({"type": "audio", "blob": part["input_audio"]["data"]})
  return {"role": msg.role, "content": parts}


def build_app(engine: Any, config: base.RunnerConfig) -> fastapi.FastAPI:
  """Builds a FastAPI app that wraps the LiteRT LM engine.

  Constructs a lightweight server exposing OpenAI-compatible HTTP endpoints
  for chat completions and scoring using the provided LiteRT LM engine.

  Args:
      engine: The initialized LiteRT LM engine instance used for inference.
      config: Configuration object or module exposing runner properties.

  Returns:
      A configured FastAPI application instance exposing the API routes.
  """

  app = fastapi.FastAPI()
  model_name = getattr(config, "model_name", "litert-model")
  always_return_not_greedy = getattr(config, "always_return_not_greedy", True)

  @app.get("/health")
  def health() -> dict[str, str]:
    """Endpoint to check if the server is healthy."""
    return {"status": "ok"}

  @app.post(f"/{api_constants.CHAT_SCORE_ENDPOINT}")
  def chat_score(req: _ChatScoreRequest):
    """Scores a chat continuation using the LiteRT LM engine."""
    # Convention: Last message is the target sequence to score (must be
    # role=assistant). All preceding messages are treated as context. The server
    # applies its own Jinja templates.
    if len(req.messages) < 2:
      raise fastapi.HTTPException(
          status_code=400,
          detail=(
              "messages must contain at least one context message and one"
              " assistant continuation"
          ),
      )
    if req.messages[-1].role != "assistant":
      raise fastapi.HTTPException(
          status_code=400,
          detail=(
              "last message must be role=assistant (the continuation to score)"
          ),
      )
    context_msgs = [_to_litert_lm_message(m) for m in req.messages[:-1]]
    continuation = req.messages[-1].content
    if not isinstance(continuation, str):
      raise fastapi.HTTPException(
          status_code=400,
          detail="continuation (last assistant message) must be a plain string",
      )
    return _chat_score(
        engine,
        context_msgs,
        continuation,
        model_name,
        always_return_not_greedy,
    )

  @app.post(f"/{api_constants.CHAT_COMPLETIONS_ENDPOINT}")
  def chat_completions(req: _ChatRequest) -> dict[str, Any]:
    """Endpoint for generating chat completions using the engine conversation API.

    This endpoint follows the OpenAI chat completions API schema.

    All generation goes through the higher-level `Conversation` API
    (`engine.create_conversation` + `conv.send_message`) — text and
    multimodal alike. The `Conversation` API does not currently expose a
    per-request token-budget hook; `_ChatRequest.max_tokens` is accepted on the
    wire but NOT ENFORCED by this endpoint. Long-form generation may run to
    EOS or to the engine's `max_num_tokens` ceiling.

    Args:
        req: The chat completion request containing the model, messages, and
          other generation parameters.

    Returns:
        A dictionary representing the completion response following the OpenAI
        schema.
    """
    stop_seqs = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    messages = [_to_litert_lm_message(m) for m in req.messages]

    # Pass preface messages directly into create_conversation and send the last
    # one as a query.
    if len(messages) > 1:
      preface = messages[:-1]
      query = messages[-1]
    else:
      preface = None
      query = messages[0] if messages else {"role": "user", "content": ""}

    sampler_config = litert_lm.SamplerConfig(temperature=req.temperature)
    with engine.create_conversation(
        messages=preface, sampler_config=sampler_config
    ) as conv:
      # Send the query message to generate the completion.
      response = conv.send_message(query)
      text = ""
      # Parse the response, which could be a plain string or a list of
      # multimodal content parts (e.g. text, images).
      if isinstance(response, dict) and "content" in response:
        content_items = response["content"]
        if isinstance(content_items, list):
          # If it is a list, filter and join all text parts.
          text = "".join([
              it.get("text", "")
              for it in content_items
              if it.get("type") == "text"
          ])
        else:
          # Otherwise treat the content directly as a string.
          text = str(content_items)
        # Truncate at the earliest stop sequence, if any.
        indices = [text.index(s) for s in stop_seqs if s in text]
        if indices:
          text = text[: min(indices)]

    return {
        "id": "chatcmpl-generation",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        # LiteRT LM currently doesn't natively expose token counts in its
        # completion responses in a format matching OpenAI's schema, so we
        # return dummy values to satisfy strict clients that require these keys
        # to exist.
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

  return app


def _render_chat_score_context(
    engine: Any, context_msgs: list[dict[str, Any]]
) -> str:
  """Renders a chat-score context using the engine's conversation API.

  Calling `conversation.render_message_to_string` per message and
  concatenating produces structurally invalid prompts: the engine's template
  treats every render as if it were the final message in the conversation
  and appends `<|im_end|>` + generation-prompt boilerplate after each one.
  Concatenating multiple such renders duplicates the generation prompt and
  loses earlier-turn structure.

  Instead, preload all-but-the-last message via `create_conversation
  (messages=...)` and render only the final message — the engine then
  templates the full conversation in one shot with the generation prompt at
  the end.

  Args:
      engine: The LiteRT LM engine instance.
      context_msgs: The conversation context (excluding the continuation that
        will be scored). Must contain at least one message.

  Returns:
      The rendered context string, ending with the model's
      add-generation-prompt suffix.
  """
  if not context_msgs:
    return ""

  with engine.create_conversation(messages=context_msgs[:-1]) as conversation:
    return conversation.render_message_to_string(context_msgs[-1])


def _chat_score(
    engine: Any,
    context_msgs: list[dict[str, Any]],
    continuation: str,
    model_name: str,
    always_return_not_greedy: bool = False,
) -> dict[str, Any]:
  """Computes scoring and token-level log probabilities for a given continuation.

  Renders the conversational context into a formatted string using the engine's
  message template, prefilling a new session with it. Then runs text scoring on
  the target continuation string to obtain the sequence score and individual
  token log probabilities. Also performs a greedy decode to determine if the
  provided continuation aligns with the greedy generation path.

  Args:
      engine: The LiteRT LM engine instance used for prefilling and scoring.
      context_msgs: A list of dictionaries representing previous chat messages,
        which serve as the conversation history/context.
      continuation: The assistant's response string whose log probabilities and
        score are being evaluated.
      model_name: The identifier of the model, to be included in the output.
      always_return_not_greedy: If True, bypasses the generation step and always
        sets is_greedy to False.

  Returns:
      A dictionary adhering to the chat score payload format containing the
      overall sequence score, token-level logprobs, and whether the output
      follows a greedy decoding path.
  """
  # Render the full conversation through the engine's chat template.
  context_str = _render_chat_score_context(engine, context_msgs)
  # The chat template is applied by the conversation API, so we disable it here.
  with engine.create_session(apply_prompt_template=False) as session:
    session.run_prefill([context_str])
    result = session.run_text_scoring([continuation])

  # `token_scores` is a list of lists, whose length matches the number of
  # candidates, which is 1 in our case.
  token_logprobs = list(result.token_scores[0]) if result.token_scores else []
  # `score` is a list of floats, whose length matches the number of candidates,
  # which is 1 in our case.
  score = result.scores[0] if result.scores else 0.0

  # Using session.run_decode to check is_greedy is a slow workaround
  # since it does full generation. We should ideally get this efficiently
  # from a token-level API directly during run_text_scoring.
  # Similar to the above, the chat template is applied by the conversation API,
  # so we disable it here.
  is_greedy = False
  # Optionally bypass full sequence generation entirely to avoid heavy latency
  # costs during scoring.
  if not always_return_not_greedy:
    with engine.create_session(apply_prompt_template=False) as session:
      session.run_prefill([context_str])
      response = session.run_decode()
    if response.texts:
      is_greedy = response.texts[0].startswith(continuation)

  return {
      "object": "chat.score",
      "model": model_name,
      "choices": [{
          "index": 0,
          "score": score,
          "logprobs": {
              "token_logprobs": token_logprobs,
              "is_greedy": is_greedy,
          },
      }],
  }


def wait_for_server(base_url: str, timeout: int = 30) -> None:
  """Polls the server's health endpoint until it is ready or times out."""
  start_time = time.time()
  while time.time() - start_time < timeout:
    try:
      response = requests.get(f"{base_url}/health", timeout=1)
      if response.status_code == 200:
        return
    except requests.exceptions.RequestException:
      pass
    time.sleep(0.5)
  raise TimeoutError(f"Server at {base_url} did not start within {timeout}s")
