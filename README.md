# AI Edge Eval

[![PyPI version](https://img.shields.io/pypi/v/ai-edge-eval.svg)](https://pypi.org/project/ai-edge-eval/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`ai-edge-eval` is an evaluation framework and CLI runner for LiteRT LM models and standard native models (e.g., HuggingFace, OpenAI), supporting both single and multi-modality use cases.

## Installation

We support installation using either `uv` (recommended for ultra-fast dependency resolution) or standard `pip` within a virtual environment (Python 3.10+).

### Option 1: Use UV (Recommended)

`uv` is an extremely fast Python package manager written in Rust.

#### 1. Create and Activate Virtual Environment

```bash
# Create a virtual environment with Python 3.13 in the current directory.
uv venv --clear --python=3.13 --seed
source .venv/bin/activate
```

#### 2. Install ai-edge-eval

**2a. Install from PyPI**

```bash
# Install the package into the active virtual environment
uv pip install -q ai-edge-eval
```

**2b. Or Install from Local Clone (Recommended for Development)**

```bash
git clone https://github.com/google-ai-edge/eval.git
cd eval

# Install in editable mode inside the active virtual environment
uv pip install -e .
```

### Option 2: Use Standard Pip

#### 1. Create and Activate Virtual Environment

```bash
# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install ai-edge-eval

**2a. Install from PyPI**

```bash
pip install -q ai-edge-eval
```

**2b. Or Install from Local Clone**

```bash
git clone https://github.com/google-ai-edge/eval.git
cd eval

# Install in editable mode
pip install -e .
```

### Optional Dependency Groups

The base installation bundles full support for LiteRT-LM evaluation out-of-the-box.

To install support for running native PyTorch/HuggingFace models, specify the optional dependency groups:

#### Using UV (Recommended)

```bash
# Install HuggingFace native runner support (includes PyTorch)
uv pip install ai-edge-eval[hf]

# Install HuggingFace multimodal runner support (includes TorchVision)
uv pip install ai-edge-eval[hf-multimodal]

# Install everything for local evaluation
uv pip install ai-edge-eval[all]
```

#### Using Standard Pip

```bash
# Install HuggingFace native runner support (includes PyTorch)
pip install ai-edge-eval[hf]

# Install HuggingFace multimodal runner support (includes TorchVision)
pip install ai-edge-eval[hf-multimodal]

# Install everything for local evaluation
pip install ai-edge-eval[all]
```

## Running Evaluations

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

Prepare your evaluation dataset in JSON Lines (`.jsonl`) format, where each entry separates the input context (`messages`) and the expected output (`ground_truth`), along with optional `metadata`. The `messages` field strictly follows the canonical OpenAI Chat Completion format (a list of dictionaries specifying `role` and `content`):

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

To run custom evaluation benchmarks, register your generation parameters and evaluation hooks via a Python file (`register_custom_tasks.py`):

```python
# File: register_custom_tasks.py

from ai_edge_eval.config.generation_config import GenerationConfig
from ai_edge_eval.custom_tasks.base import CustomTask, DatasetRow
from ai_edge_eval.custom_tasks.registry import TaskRegistry

from typing import Iterator

def exact_match(
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
    metric_fn=exact_match,
    generation_config=GenerationConfig(
        temperature=0.5, max_new_tokens=64, stop_sequences=["\n"]
    )
)

TaskRegistry.global_registry().register(qa_task)
```

### 3. Run Custom Evaluation

Point the CLI to your custom registration file authored in Step 2 using the `--custom-tasks-file` flag:

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

---

## Dataset Licensing and Terms of Use

`ai-edge-eval` is an evaluation runner and command-line toolkit licensed under the Apache 2.0 License.

### Third-Party Dataset Integration
When executing benchmark evaluations, `ai-edge-eval` relies on upstream execution frameworks (such as EleutherAI's `lm-eval` harness) to dynamically download and cache evaluation datasets from external sources (e.g., HuggingFace Hub).

**`ai-edge-eval` does not host, redistribute, or sublicense these external datasets.**

### User Responsibility
Every evaluation dataset maintains its own licensing terms, ownership rights, and permitted usage policies (including potential non-commercial restrictions). 

**By executing evaluations using `ai-edge-eval`, you are responsible for:**
1. Reviewing and consenting to the specific terms of service and license agreement associated with each evaluated benchmark.
2. Adhering to any commercial or distribution constraints associated with the underlying data.

For detailed licensing information regarding specific datasets, refer to their respective model and dataset cards on the HuggingFace Hub or official repository pages.
