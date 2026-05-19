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

r"""Example script demonstrating how to register a custom evaluation task.

Example command:
```bash
export GEMINI_API_KEY="your_actual_api_key_here"
ai-edge-eval \
  --framework custom \
  --runner litert-lm \
  --model-path /path/to/your_model.litertlm \
  --custom-tasks-file pirate_gemma_custom_task.py \
  --tasks pirate_gemma_eval \
  --output-dir /tmp/results
```
"""

import json
import logging
import os
from typing import Iterator

from model_eval import custom_tasks
import datasets
from google import genai
from google.genai import types
import tqdm

# Suppress verbose INFO logging from httpx/httpcore and google.genai (e.g., AFC
# logs)
for logger_name in [
    "httpx",
    "httpcore",
    "google",
    "google.genai",
    "google_genai",
    "google_genai.models",
]:
  logger = logging.getLogger(logger_name)
  logger.setLevel(logging.WARNING)
  logger.propagate = False

# Also suppress via Logger manager dict to catch any dynamically created loggers
for name, logger_obj in logging.Logger.manager.loggerDict.items():
  if isinstance(logger_obj, logging.Logger) and any(
      sub in name for sub in ["httpx", "httpcore", "google", "genai"]
  ):
    logger_obj.setLevel(logging.WARNING)
    logger_obj.propagate = False


# ---------------------------------------------------------------------------
# 1. Dataset Generator
# ---------------------------------------------------------------------------
def pirate_dataset_generator():
  """Yields DatasetRows from the HuggingFace dataset."""
  # Note: The erintwalsh/pirate-gemma-tutorial dataset only has a 'train' split.
  hf_dataset = datasets.load_dataset(
      "erintwalsh/pirate-gemma-tutorial", split="train"
  )

  for row in hf_dataset:
    # The dataset uses 'user' for the prompt and 'pirate' for the ground truth
    prompt_text = row.get("user", "")
    expected = row.get("pirate", "")

    yield custom_tasks.DatasetRow(
        messages=[{"role": "user", "content": prompt_text}],
        ground_truth=expected,
    )


# ---------------------------------------------------------------------------
# 2. LLM-as-a-Judge Metric Function
# ---------------------------------------------------------------------------
def llm_judge_metric_fn(
    predictions: Iterator[str],
    ground_truths: Iterator[str],
    rows: Iterator[custom_tasks.DatasetRow],
) -> dict[str, float]:
  """Uses Gemini to evaluate predictions on Pirate Persona and Content Relevance."""
  api_key = os.environ.get("GEMINI_API_KEY")
  if not api_key:
    raise ValueError(
        "Please set the GEMINI_API_KEY environment variable before running the"
        " evaluation:\n  export GEMINI_API_KEY='your_api_key_here'\n "
        " ai-edge-eval --framework custom ..."
    )
  client = genai.Client(api_key=api_key)

  total_pirate_score = 0
  total_helpfulness = 0
  count = 0

  pred_list = list(predictions)
  gt_list = list(ground_truths)
  row_list = list(rows)

  for pred, gt, row in tqdm.tqdm(
      zip(pred_list, gt_list, row_list),
      total=len(pred_list),
      desc="LLM Judge Evaluation",
  ):
    user_prompt = row["messages"][-1]["content"]

    eval_prompt = f"""
        You are an expert evaluator of language models. Your task is to evaluate the quality of a model's response based on a specific persona constraint. 
        The model was asked to answer a prompt while acting like a pirate.

        User Prompt: {user_prompt}
        Expected Answer (Ground Truth): {gt}
        Model Response: {pred}

        Please evaluate the model's response based on the following two criteria:

        1. Pirate Persona (Score 0-5): Does the response authentically sound like a pirate? 
           - 5: Perfect pirate tone, uses rich pirate vocabulary, and maintains the persona flawlessly.
           - 0: Completely ignores the persona, sounds like a standard AI assistant.

        2. Content Relevance (Score 0-5): How well does the core information in the model's response match the Expected Answer?
           - 5: Conveys the same core meaning and facts as the Expected Answer, regardless of the exact phrasing.
           - 0: Completely ignores the expected meaning, providing an irrelevant or nonsensical answer.

        Provide your evaluation in strict JSON format as follows, with no markdown formatting or extra text:
        {{"pirate_score": <int>, "content_relevance_score": <int>, "reasoning": "<string>"}}
        """

    try:
      response = client.models.generate_content(
          model="gemini-3.1-pro-preview",
          contents=eval_prompt,
          config=types.GenerateContentConfig(
              response_mime_type="application/json"
          ),
      )

      result = json.loads(response.text)
      total_pirate_score += result.get("pirate_score", 0)
      total_helpfulness += result.get("content_relevance_score", 0)
      count += 1

    except Exception as e:  # pylint: disable=broad-except
      print(f"Error evaluating row: {e}")

  if count == 0:
    return {"avg_pirate_score": 0.0, "avg_content_relevance": 0.0}

  return {
      "avg_pirate_score": total_pirate_score / count,
      "avg_content_relevance": total_helpfulness / count,
  }


# ---------------------------------------------------------------------------
# 3. Task Registration
# ---------------------------------------------------------------------------
pirate_task = custom_tasks.CustomTask(
    name="pirate_gemma_eval",
    dataset=pirate_dataset_generator,
    metric_fn=llm_judge_metric_fn,
)

custom_tasks.TaskRegistry.global_registry().register(pirate_task)
