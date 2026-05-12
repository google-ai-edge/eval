# AI Edge Eval Guide

This repository provides an evaluation framework and CLI runner for LiteRT LM models as well as standard native models (e.g. HF, OpenAI) supporting both single and multi-modality use cases.

## Running Evaluations in Google3

### Custom Runner (LiteRT LM - Text Sampling)

To run evaluation on a standard text benchmark like `ifeval` and `bbh`:

```bash
ai-edge-eval \
      --runner litert-lm \
      --model-path /path/to/model.litertlm \
      --device cpu \
      --tasks ifeval \
      --tasks bbh \
      --framework lm-eval \
      --limit 10 \
      --output-dir your_result_directory
```

### Custom Runner (LiteRT LM - Text Scoring)

To run evaluation on a standard multiple-choice scoring benchmark like `piqa`:

```bash
ai-edge-eval \
      --runner litert-lm \
      --model-path /path/to/model.litertlm \
      --device cpu \
      --tasks piqa \
      --framework lm-eval \
      --limit 10 \
      --output-dir your_result_directory
```

### Custom Runner (LiteRT LM - Multimodal Sampling)

To run multimodal sampling using vision capabilities (e.g., on `mmmu_val`):

```bash
ai-edge-eval \
      --runner litert-lm \
      --model-path /path/to/model.litertlm \
      --device cpu \
      --runner-args "vision_backend=cpu" \
      --tasks mmmu_val \
      --framework lm-eval \
      --limit 10 \
      --output-dir your_result_directory
```

### Direct Native Library (HuggingFace, etc)

To run evaluation natively using direct library wrappers via lm-eval:

```bash
ai-edge-eval \
      --runner hf \
      --model-path huggingface/repo \
      --device cpu \
      --tasks mmlu \
      --framework lm-eval \
      --limit 10 \
      --output-dir your_result_directory
```

### Direct Native Library (HuggingFace Multimodal)

To run multimodal evaluation natively using direct library wrappers via lm-eval:

```bash
ai-edge-eval \
      --runner hf-multimodal \
      --model-path huggingface/repo \
      --device cpu \
      --tasks mmmu_val \
      --framework lm-eval \
      --limit 10 \
      --batch-size 1 \
      --output-dir your_result_directory
```

**Note:** For HuggingFace runners, `huggingface/repo` refers to the HuggingFace model ID, such as `Qwen/Qwen2.5-7B-Instruct` or `google/gemma-3-270m`.

---

## Custom Task CUJ

### 1. Prepare the Dataset

Prepare your evaluation dataset in JSON Lines (`.jsonl`) format, where each entry separates the input context (`messages`) and the expected output (`ground_truth`), along with optional `metadata`:

```json
{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "ground_truth": "Paris"
}
{
  "messages": [{"role": "user", "content": "Calculate 5 + 7"}],
  "ground_truth": "12"
}
```

### 2. Task Definition

To run custom evaluation benchmarks, register your generation parameters and evaluation hooks via a Python file:

```python
from ai_edge_eval.config.generation_config import GenerationConfig
from ai_edge_eval.custom_tasks.base import CustomTask, DatasetRow
from ai_edge_eval.custom_tasks.registry import TaskRegistry

from typing import Iterator

def dummy_exact_match(
    preds: Iterator[str], gts: Iterator[str], rows: Iterator[DatasetRow[str]]
) -> dict[str, float]:
  # Retrieve generated text and ground truth text.
  p = [text.strip().lower() for text in preds]
  g = [text.strip().lower() for text in gts]
  accuracy = sum(pi == gi for pi, gi in zip(p, g)) / len(p)
  return {"exact_match": accuracy}

qa_task = CustomTask(
    name="my_custom_qa",
    dataset="path/to/dataset.jsonl",
    metric_fn=dummy_exact_match,
    generation_config=GenerationConfig(
        temperature=0.5, max_new_tokens=64, stop_sequences=["\n"]
    )
)

TaskRegistry.global_registry().register(qa_task)
```

### 3. Run Custom Evaluation

Point the CLI to your custom registration file using the `--custom-tasks-file` flag:

```bash
ai-edge-eval \
      --runner litert-lm \
      --runner-args "model_path=/path/to/model.litertlm,backend=cpu" \
      --tasks my_custom_qa \
      --framework custom \
      --custom-tasks-file register_custom_tasks.py \
      --eval-args "limit=10" \
      --output-dir your_result_directory
```

---

## Discovery Commands

### Argument Discovery

You can use the `list-args` subcommand to inspect the available configurations and parameters exposed by a given runner or evaluation framework:

```bash
# Discover runner arguments
ai-edge-eval list-args --runner litert-lm

# Discover evaluation framework arguments
ai-edge-eval list-args --framework lm-eval
```

### Supported Tasks and Runners

You can use the `list-tasks` and `list-runners` subcommands to view the allowlist of supported tasks and runners for a given framework:

```bash
# List supported tasks for a framework
ai-edge-eval list-tasks --framework lm-eval

# List supported runners for a framework
ai-edge-eval list-runners --framework lm-eval
```
