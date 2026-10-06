# 🎯 AI Edge Eval

**An advanced evaluation framework and CLI runner for LiteRT LM and native models.**

[![PyPI version](https://img.shields.io/pypi/v/ai-edge-eval.svg)](https://pypi.org/project/ai-edge-eval/)
[![Python Support](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/ai-edge-eval/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

`ai-edge-eval` is a powerful evaluation framework and CLI runner designed for **LiteRT LM models** and standard native models (e.g., **HuggingFace**). Built for POSIX-compliant systems, it officially supports **Linux**, **macOS**, and **Windows (via WSL2)**, providing robust support for both **single-modality** (text) and **multi-modality** (vision + text) use cases.

## 📖 Table of Contents
- [🚀 Installation](#-installation)
  - [📋 System Requirements](#-system-requirements)
  - [Option 1: Use uv (Recommended)](#option-1-use-uv-recommended)
  - [Option 2: Use Standard pip](#option-2-use-standard-pip)
  - [Optional Dependency Groups](#optional-dependency-groups)
- [⚡ Running Evaluations](#-running-evaluations)
  - [LiteRT LM Runners](#🤖-litert-lm-runners)
  - [Direct Native Library Runners](#🚀-direct-native-library-runners-huggingface-etc)
  - [HTTP Server Runner](#🌐-http-server-runner)
  - [Lighteval Framework](#🪶-lighteval-framework)
  - [Subsetting and Slicing Datasets](#✂️-subsetting-and-slicing-datasets)
- [🛠️ Custom Task CUJ](#️-custom-task-cuj)
  - [1. Prepare the Dataset](#1-prepare-the-dataset)
  - [2. Task Definition](#2-task-definition)
  - [3. Run Custom Evaluation](#3-run-custom-evaluation)
- [🔍 Discovery Commands](#-discovery-commands)
- [⚖️ Dataset Licensing and Terms of Use](#️-dataset-licensing-and-terms-of-use)

---

## 🚀 Installation

### 📋 System Requirements

`ai-edge-eval` requires a POSIX-compliant Unix-like environment. The following platforms are officially supported:
- **Linux**: Standard distributions (e.g., Ubuntu, Debian).
- **macOS**: Both Intel and Apple Silicon (M-series) architectures.
- **Windows**: Supported **exclusively** via the [Windows Subsystem for Linux (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/install). *(Note: Native Windows execution via CMD or PowerShell is not supported.)*

---

We support installation using either `uv` (recommended for ultra-fast dependency resolution) or standard `pip` within a virtual environment (Python 3.10+).

### Option 1: Use `uv` (Recommended)

> [!TIP]
> [`uv`](https://github.com/astral-sh/uv) is an extremely fast Python package manager written in Rust. Using it significantly speeds up environment creation and dependency installation.

#### 1. Create and Activate Virtual Environment

```bash
# Create a virtual environment with Python 3.13 in the current directory.
uv venv --clear --python=3.13 --seed
source .venv/bin/activate
```

#### 2. Install `ai-edge-eval`

**Option A: Install from PyPI**
```bash
# Install the package into the active virtual environment
uv pip install -q ai-edge-eval
```

**Option B: Install from Local Clone (Recommended for Development)**
```bash
git clone https://github.com/google-ai-edge/eval.git
cd eval

# Install in editable mode inside the active virtual environment
uv pip install -e .
```

### Option 2: Use Standard `pip`

#### 1. Create and Activate Virtual Environment

```bash
# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install `ai-edge-eval`

**Option A: Install from PyPI**
```bash
pip install -q ai-edge-eval
```

**Option B: Install from Local Clone**
```bash
git clone https://github.com/google-ai-edge/eval.git
cd eval

# Install in editable mode
pip install -e .
```

---

### 📦 Optional Dependency Groups

The base installation bundles full support for LiteRT-LM evaluation out-of-the-box. To install support for running native PyTorch/HuggingFace models, specify the optional dependency groups:

#### Using `uv` (Recommended)
```bash
# Install HuggingFace native runner support (includes PyTorch)
uv pip install "ai-edge-eval[hf]"

# Install HuggingFace multimodal runner support (includes TorchVision)
uv pip install "ai-edge-eval[hf-multimodal]"

# Install the Lighteval evaluation framework (alternative to the default lm-eval)
uv pip install "ai-edge-eval[lighteval]"

# Install everything for local evaluation
uv pip install "ai-edge-eval[all]"
```

#### Using Standard `pip`
```bash
# Install HuggingFace native runner support (includes PyTorch)
pip install "ai-edge-eval[hf]"

# Install HuggingFace multimodal runner support (includes TorchVision)
pip install "ai-edge-eval[hf-multimodal]"

# Install the Lighteval evaluation framework (alternative to the default lm-eval)
pip install "ai-edge-eval[lighteval]"

# Install everything for local evaluation
pip install "ai-edge-eval[all]"
```

> [!NOTE]
> Quotes around package names with brackets (e.g., `"ai-edge-eval[hf]"`) prevent shell globbing issues in Zsh and Bash.

---

## ⚡ Running Evaluations

`ai-edge-eval` provides high-performance runners for both LiteRT models and native HuggingFace models.

### 🤖 LiteRT LM Runners

#### Text Sampling
Run evaluation on standard text benchmarks like `ifeval` and `bbh`:

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

#### Text Scoring
Run evaluation on standard multiple-choice scoring benchmarks like `piqa`:

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

#### Multimodal Sampling
Run multimodal sampling using vision capabilities (e.g., on `mmmu_val`):

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

### 🚀 Direct Native Library Runners (HuggingFace, etc.)

#### Text Evaluation
Run evaluation natively using direct library wrappers via `lm-eval`:

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

#### Multimodal Evaluation
Run multimodal evaluation natively using direct library wrappers via `lm-eval`:

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

> [!IMPORTANT]
> For HuggingFace runners, `huggingface/repo` refers to the HuggingFace model ID, such as `Qwen/Qwen2.5-7B-Instruct` or `google/gemma-3-270m`.

### 🌐 HTTP Server Runner

The `http-server` runner evaluates a model served by an HTTP server that is
already running, regardless of the backend or where it runs (e.g., a phone,
another machine, or a cloud instance). Unlike the local runners, it does not
manage the model lifecycle: it provides transport (`server_url`, optional
`request_timeout_sec`, and arbitrary `server_args`), while the evaluation
framework or task determines which endpoint(s) to call:

| Framework / task type | Endpoint(s) called on the server |
| :--- | :--- |
| `lm-eval` or `lighteval`, generation tasks (e.g., `ifeval`) | `POST /v1/chat/completions` (OpenAI-compatible) |
| `lm-eval` or `lighteval`, scoring tasks (e.g., `piqa`, `arc:easy`) | `POST /v1/chat/score` (served by the LiteRT-LM server; not part of the OpenAI API) |
| `custom` | `POST /<task.endpoint>`, once per entry of each row's `requests` |

```bash
ai-edge-eval \
      --runner http-server \
      --runner-args "server_url=http://<address>:<port>,model_name=<model_name>" \
      --tasks ifeval \
      --framework lm-eval \
      --limit 2 \
      --output-dir your_result_directory
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `server_url` | **Required.** The server's base URL, without any endpoint path (e.g., `http://10.0.0.1:8080`, not `http://10.0.0.1:8080/v1/chat/completions`). | None |
| `request_timeout_sec` | Optional. Per-request HTTP timeout in seconds (supported on all server runners, including `litert-lm` and `http-server`). | None (framework default) |
| *any other key* | Optional. Kept verbatim as the runner's `server_args`; the runner itself never interprets them. Use this for values the **server** needs in requests, most commonly `model_name` when the server hosts several models. | `{}` |

Combine multiple runner arguments with commas:
`--runner-args server_url=...,model_name=...`.

The model is always chosen on the runner side. Local runners load it from
`--model-path`; the `http-server` runner has no model to load, so you name the
one the server should use via `model_name` in `--runner-args`:

| Runner | Framework | How the model is chosen | What the framework sends |
| :--- | :--- | :--- | :--- |
| `litert-lm` | `lm-eval` / `lighteval` | `--model-path` (or `model_path=` in `--runner-args`); the runner starts a server for that model | `"model": "default_model"` (ignored by the LiteRT-LM server) |
| `litert-lm` | `custom` | `--model-path` (or `model_path=` in `--runner-args`) | `task_args` holds only `--eval-args`; `chat_request(...)` sends `"model": "default_model"` (ignored by the LiteRT-LM server) |
| `http-server` | `lm-eval` / `lighteval` | `--runner-args "...,model_name=<name>"` | `"model": "<name>"` in every OpenAI request (`default_model` if omitted) |
| `http-server` | `custom` | `--runner-args "...,model_name=<name>"` | `server_args` are merged into the task's `task_args` (see [Custom Task](#-custom-task)), so the task decides how to use `model_name` |

> [!NOTE]
> The runner is stateless. Every request contains the full prompt, including
> all message history, and no reset signal is sent between examples. If your
> server uses KV caching, it must detect unrelated prompts (or empty history)
> and clear its state itself.

### 🪶 Lighteval Framework

`--framework lighteval` is an alternative evaluation framework alongside the default `lm-eval`. Install via `ai-edge-eval[lighteval]` (see [Optional Dependency Groups](#-optional-dependency-groups)).

Supports two runner paths:

- `--runner litert-lm` — scores via the LiteRT-LM server's `/v1/chat/score` endpoint (auto-launched).
- `--runner accelerate` — scores natively via lighteval's HuggingFace/Accelerate backend.

Supported tasks (from `model_eval/config/tasks.yaml`): `mmlu`, `arc:easy`, `arc:challenge`, `winogrande`, `ifeval`, `bigbench_hard`.

#### LiteRT-LM Runner

```bash
ai-edge-eval \
      --runner litert-lm \
      --model-path litert-community/SmolLM2-360M-Instruct/SmolLM2_360M_instruct.litertlm \
      --device cpu \
      --tasks arc:easy \
      --framework lighteval \
      --batch-size 1 \
      --limit 20 \
      --output-dir your_result_directory
```

#### Accelerate Runner (HuggingFace native)

```bash
ai-edge-eval \
      --runner accelerate \
      --model-path HuggingFaceTB/SmolLM2-360M-Instruct \
      --device cpu \
      --tasks arc:easy \
      --framework lighteval \
      --batch-size 1 \
      --limit 20 \
      --output-dir your_result_directory
```

> [!IMPORTANT]
> Always pin `--batch-size 1` when using `--framework lighteval`. Without it, lighteval auto-picks the largest batch that fits, which can OOM on the LiteRT-LM runner and produces padding-dependent results across runs. We recommend using a small `--limit` (e.g., `--limit 5`) or `--sample-range` (e.g., `--sample-range 0 4`) for generation and sampling tasks (`ifeval`, `bigbench_hard`) to perform quick smoke checks. This allows you to verify task configuration and gauge the overall evaluation size (including token generation volume) before launching comprehensive runs.

> [!NOTE]
> Some known cross-path differences when running the same model under both lighteval runners: (1) the accelerate path injects the tokenizer's default system message; the litert-lm path doesn't — prompts differ on tasks without an explicit system message. (2) The accelerate path is currently non-deterministic across process invocations (upstream lighteval issue); the litert-lm path is byte-deterministic.

### ✂️ Subsetting and Slicing Datasets

When debugging, performing smoke checks, or evaluating specific dataset slices, you can restrict the number of samples evaluated using the following mutually exclusive flags:

- `--limit <int | float>`: Limits evaluation to a maximum number of samples (e.g., `--limit 10`) or a fraction of the total dataset (e.g., `--limit 0.1` for 10%).
- `--sample-range <start> <end>`: Evaluates a targeted index range of samples, inclusive of both `start` and `end` (e.g., `--sample-range 10 20` evaluates samples 10 through 20).

```bash
# Example: Evaluate a specific 5-sample slice (indices 10 to 14) using lm-eval
ai-edge-eval \
      --runner litert-lm \
      --model-path /path/to/model.litertlm \
      --device cpu \
      --tasks ifeval \
      --framework lm-eval \
      --sample-range 10 14 \
      --output-dir your_result_directory
```

> [!NOTE]
> `--limit` and `--sample-range` are mutually exclusive. Specifying both will result in a configuration error.

---

## 🛠️ Custom Task CUJ

`ai-edge-eval` makes it seamless to define and run custom evaluation benchmarks
against any server endpoint.

### 1. Prepare the Dataset

Prepare your evaluation dataset in JSON Lines (`.jsonl`) format or as a Python
generator `dataset(task_args)`. Each row has `requests` (the list of JSON
request bodies to POST in order for that row) and `ground_truth`, along with
optional `metadata`. For chat-completions tasks,
`chat_request(messages, generation_config, model_name=...)` builds an
OpenAI-compatible request body and `chat_response_text(response)` extracts the
generated assistant text:

```json
{
  "requests": [{"model": "default_model", "messages": [{"role": "user", "content": "What is the capital of France?"}], "temperature": 0.5, "max_tokens": 64, "stop": ["\n"]}],
  "ground_truth": "Paris"
}
{
  "requests": [{"model": "default_model", "messages": [{"role": "user", "content": "Calculate 5 + 7"}], "temperature": 0.5, "max_tokens": 64, "stop": ["\n"]}],
  "ground_truth": "12"
}
```

### 2. Task Definition

Register your evaluation task via a Python file (e.g.,
`register_custom_tasks.py`):

```python
# File: register_custom_tasks.py

from typing import Any, Iterator
from model_eval.custom_tasks import CustomTask, DatasetRow, TaskRegistry, chat_response_text

def exact_match(
    preds: Iterator[list[Any]], gts: Iterator[str], rows: Iterator[DatasetRow[str]]
) -> dict[str, float]:
  # Retrieve generated text from each row's first response and compare with ground truth.
  p = [(chat_response_text(r[0]) or "").strip().lower() for r in preds]
  g = [text.strip().lower() for text in gts]
  accuracy = sum(pi == gi for pi, gi in zip(p, g)) / len(p)
  return {"exact_match": accuracy}

qa_task = CustomTask(
    name="my_custom_qa",
    endpoint="v1/chat/completions",
    dataset="path/to/dataset.jsonl",
    metric_fn=exact_match,
)

TaskRegistry.global_registry().register(qa_task)
```

### 3. Run Custom Evaluation

Point the CLI to your custom registration file using `--custom-tasks-file`:

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

`dataset` may also be a callable `dataset(task_args)` returning an iterator of
`DatasetRow`. `task_args` is the merge of two sources:

1. the runner's `server_args` (e.g. `model_name` from
   `--runner-args "...,model_name=gemma-4-E2B-it"` for `http-server`; local
   runners such as `litert-lm` have none), and
2. the `--eval-args` entries the framework does not consume itself (it consumes
   `limit`, `batch_size`, and `samples`), so a task can take its own
   configuration, e.g. `--eval-args split=dev`.

On a key collision `--eval-args` win. A chat task typically reads
`task_args.get("model_name")` and passes it to
`chat_request(..., model_name=...)`. File datasets (`.jsonl`/`.csv`) accept no
eval-args task args; passing any is an error (`server_args` are ignored).

---

## 🔍 Discovery Commands

`ai-edge-eval` includes built-in discovery utilities to help you explore supported configurations, tasks, and runners.

### Argument Discovery
Use the `list-args` subcommand to inspect the available configurations and parameters exposed by a given runner or evaluation framework:

```bash
# Discover runner arguments
ai-edge-eval list-args --runner litert-lm

# Discover evaluation framework arguments
ai-edge-eval list-args --framework lm-eval
```

### Supported Tasks and Runners
Use the `list-tasks` and `list-runners` subcommands to view the allowlist of supported tasks and runners for a given framework:

```bash
# List supported tasks for a framework
ai-edge-eval list-tasks --framework lm-eval

# List supported runners for a framework
ai-edge-eval list-runners --framework lm-eval
```

> [!NOTE]
> **Subtask Allowlisting**: Listing a parent task (e.g., `mmlu`, `bbh`) in the task allowlist implicitly authorizes all of its concrete subtasks. You can run individual subtasks (e.g., `--tasks mmlu_abstract_algebra` for `lm-eval` or `--tasks mmlu:abstract_algebra` for `lighteval`) without needing to list every subtask explicitly.
>
> To see the expanded list of subtasks for each parent task in the allowlist, use the `--show-subtasks` flag:
> ```bash
> ai-edge-eval list-tasks --framework lm-eval --show-subtasks
> ```

---

## ⚖️ Dataset Licensing and Terms of Use

`ai-edge-eval` is an evaluation runner and command-line toolkit licensed under the **Apache 2.0 License**.

### Third-Party Dataset Integration

> [!WARNING]
> When executing benchmark evaluations, `ai-edge-eval` relies on upstream execution frameworks (such as EleutherAI's `lm-eval` harness) to dynamically download and cache evaluation datasets from external sources (e.g., HuggingFace Hub).
> **`ai-edge-eval` does not host, redistribute, or sublicense these external datasets.**

### User Responsibility

Every evaluation dataset maintains its own licensing terms, ownership rights, and permitted usage policies (including potential non-commercial restrictions).

> [!IMPORTANT]
> **By executing evaluations using `ai-edge-eval`, you are responsible for:**
> 1. Reviewing and consenting to the specific terms of service and license agreement associated with each evaluated benchmark.
> 2. Adhering to any commercial or distribution constraints associated with the underlying data.

For detailed licensing information regarding specific datasets, refer to their respective model and dataset cards on the HuggingFace Hub or official repository pages.
