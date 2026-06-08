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

Example execution command:
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
"""

import base64
import io
from typing import Iterator
import wave

from model_eval import config
from model_eval import custom_tasks


def _silence_wav(seconds: float = 1.0, sr: int = 16000) -> bytes:
  """Generates dummy synthetic silence WAV audio bytes."""
  num_samples = int(seconds * sr)
  buf = io.BytesIO()
  with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(b"\x00" * (num_samples * 2))
  return buf.getvalue()


def asr_dataset() -> Iterator[custom_tasks.DatasetRow]:
  """Yields dataset samples formatted with audio input messages and reference ground truth."""
  for gold in (
      "speech recognition validation test",
      "artificial intelligence at the edge",
      "verbatim audio transcription sample",
  ):
    wav_bytes = _silence_wav()
    encoded_audio = base64.b64encode(wav_bytes).decode("utf-8")
    yield {
        "messages": [{
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
                    "text": "Transcribe this audio verbatim.",
                },
            ],
        }],
        "ground_truth": gold,
    }


def asr_metrics(
    preds: Iterator[str],
    gts: Iterator[str],
    rows: Iterator[custom_tasks.DatasetRow],
) -> dict[str, float]:
  """Evaluates speech recognition output matching rates and response completeness."""
  del rows
  preds_list = [p.strip().lower() for p in preds]
  gts_list = [g.strip().lower() for g in gts]

  if not preds_list:
    return {
        "non_empty_rate": 0.0,
        "exact_match": 0.0,
        "substring_match": 0.0,
    }

  non_empty_count = sum(1 for p in preds_list if bool(p))
  exact_match_count = sum(p == g for p, g in zip(preds_list, gts_list))
  substring_match_count = sum(g in p for p, g in zip(preds_list, gts_list))

  total = len(preds_list)
  return {
      "non_empty_rate": non_empty_count / total,
      "exact_match": exact_match_count / total,
      "substring_match": substring_match_count / total,
  }


asr_task = custom_tasks.CustomTask(
    name="asr_eval",
    dataset=asr_dataset,
    metric_fn=asr_metrics,
    generation_config=config.GenerationConfig(
        temperature=0.0, max_new_tokens=64
    ),
)

custom_tasks.TaskRegistry.global_registry().register(asr_task)
