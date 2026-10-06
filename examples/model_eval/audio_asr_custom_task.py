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

r"""Example script defining an Automatic Speech Recognition (ASR) custom task.

Example execution command against an already running OpenAI-compatible server:
```bash
ai-edge-eval \
  --framework custom \
  --runner http-server \
  --runner-args server_url=http://localhost:8080,model_name=gemma-4-E2B-it \
  --custom-tasks-file examples/model_eval/audio_asr_custom_task.py \
  --tasks asr_eval \
  --limit 3 \
  --batch-size 1 \
  --output-dir /tmp/audio_results
```

Or with the in-process LiteRT-LM runner:
```bash
ai-edge-eval \
  --framework custom \
  --runner litert-lm \
  --model-path /path/to/audio_capable.litertlm \
  --runner-args audio_backend=cpu \
  --custom-tasks-file examples/model_eval/audio_asr_custom_task.py \
  --tasks asr_eval \
  --limit 3 \
  --batch-size 1 \
  --output-dir /tmp/audio_results
```

Because each request embeds the audio as base64, the task sets
`sample_input_fn` so the per-sample outputs record a small summary of the
input instead of the full request payloads.
"""

import base64
import io
from typing import Any, Iterator
import wave

from model_eval import config
from model_eval import custom_tasks

_GENERATION_CONFIG = config.GenerationConfig(temperature=0.0, max_new_tokens=64)
_PROMPT = "Transcribe this audio verbatim."
_AUDIO_SECONDS = 1.0


def _silence_wav(seconds: float = _AUDIO_SECONDS, sr: int = 16000) -> bytes:
  """Generates dummy synthetic silence WAV audio bytes."""
  num_samples = int(seconds * sr)
  buf = io.BytesIO()
  with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(b"\x00" * (num_samples * 2))
  return buf.getvalue()


def asr_dataset(
    task_args: custom_tasks.TaskArgs | None = None,
) -> Iterator[custom_tasks.DatasetRow]:
  """Yields dataset samples formatted with audio input requests and reference ground truth."""
  model_name = (task_args or {}).get("model_name")
  for index, gold in enumerate((
      "speech recognition validation test",
      "artificial intelligence at the edge",
      "verbatim audio transcription sample",
  )):
    wav_bytes = _silence_wav()
    encoded_audio = base64.b64encode(wav_bytes).decode("utf-8")
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": encoded_audio,
                    "format": "wav",
                },
            },
            {
                "type": "text",
                "text": _PROMPT,
            },
        ],
    }]
    yield {
        "requests": [
            custom_tasks.chat_request(
                messages, _GENERATION_CONFIG, model_name=model_name
            )
        ],
        "ground_truth": gold,
        "metadata": {"id": f"utt-{index}", "audio_length_s": _AUDIO_SECONDS},
    }


def asr_sample_input(row: custom_tasks.DatasetRow) -> dict[str, Any]:
  """Summarises a row's input without the base64 audio payload."""
  return {"prompt": _PROMPT, **row.get("metadata", {})}


def asr_metrics(
    preds: Iterator[list[dict[str, Any] | None]],
    gts: Iterator[str],
    rows: Iterator[custom_tasks.DatasetRow],
    task_args: custom_tasks.TaskArgs,
) -> dict[str, float]:
  """Evaluates speech recognition output matching rates and response completeness."""
  del task_args  # Unused.
  preds_list = [
      (custom_tasks.chat_response_text(p[0]) or "").strip().lower() if p else ""
      for p in preds
  ]
  gts_list = [g.strip().lower() for g in gts]
  # The framework records per-request telemetry (including wall-clock latency)
  # in the row's metadata, so a task can report timing alongside quality
  # metrics.
  latencies = [
      row["metadata"]["telemetry"][0]["latency_sec"]
      for row in rows
      if row["metadata"]["telemetry"][0] is not None
  ]

  if not preds_list:
    return {
        "non_empty_rate": 0.0,
        "exact_match": 0.0,
        "substring_match": 0.0,
        "mean_latency_sec": 0.0,
    }

  non_empty_count = sum(1 for p in preds_list if bool(p))
  exact_match_count = sum(p == g for p, g in zip(preds_list, gts_list))
  substring_match_count = sum(g in p for p, g in zip(preds_list, gts_list))

  total = len(preds_list)
  return {
      "non_empty_rate": non_empty_count / total,
      "exact_match": exact_match_count / total,
      "substring_match": substring_match_count / total,
      "mean_latency_sec": sum(latencies) / len(latencies) if latencies else 0.0,
  }


asr_task = custom_tasks.CustomTask(
    name="asr_eval",
    endpoint="v1/chat/completions",
    dataset=asr_dataset,
    metric_fn=asr_metrics,
    sample_input_fn=asr_sample_input,
)

custom_tasks.TaskRegistry.global_registry().register(asr_task)
