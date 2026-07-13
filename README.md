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

`ai-edge-eval` makes it seamless to define and evaluate custom evaluation benchmarks tailored to your specific datasets and metrics. You can author custom tasks using either **Declarative Catalogs (YAML)** or **Pure Python Scripting**.

### Approach 1: Declarative Catalogs (YAML)

For highly structured projects or when maintaining large suites of evaluations, we recommend separating reusable data loading and metric calculation machinery from configuration.

#### 1. Define Machinery in Python
Register reusable dataset loaders and evaluation metrics by name in a Python script (e.g., `plugins.py`):

```python
# File: plugins.py

import csv
from model_eval.custom_tasks import register_loader, register_metric, DatasetRow

@register_loader("my_csv_loader")
def load_csv(spec) -> list[DatasetRow]:
  # Parse dataset row elements based on options configured in the YAML spec.
  rows = []
  with open(spec.options["file"], "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
      rows.append(DatasetRow(messages=[{"role": "user", "content": r["prompt"]}], ground_truth=r["target"]))
  return rows

@register_metric("my_accuracy")
def eval_accuracy(preds, gts, rows, *, normalizer) -> dict[str, float]:
  p = [normalizer(text) for text in preds]
  g = [normalizer(text) for text in gts]
  return {"accuracy": sum(pi == gi for pi, gi in zip(p, g)) / len(p)}
```

#### 2. Author Task Spec in YAML
Declare your evaluation benchmark tasks in a YAML file (e.g., `tasks.yaml`). Task names use colon-separated prefixes (`:`) to define natural hierarchies and groups.

> [!IMPORTANT]
> **Required Generation Config**: The `generation_config` block is strictly required for all task specifications. If you want to use the default settings (e.g., default temperature, token limits), you must still provide an empty block: `generation_config: {}`.

```yaml
# File: tasks.yaml
- name: my_project:qa:dev
  loader: my_csv_loader
  metrics: [my_accuracy]
  normalizer: identity
  generation_config:
    max_new_tokens: 64
  options:
    file: path/to/dev_set.csv

- name: my_project:qa:test
  loader: my_csv_loader
  metrics: [my_accuracy]
  normalizer: identity
  generation_config:
    max_new_tokens: 64
  options:
    file: path/to/test_set.csv
```

> [!TIP]
> **Matrix Expansion**: You can easily generate combinatorial suites of tasks using the `for_each` key:
> ```yaml
> - name: my_project:math_{split}
>   loader: my_csv_loader
>   metrics: [my_accuracy]
>   generation_config: {}
>   for_each:
>     split: [dev, test]
> ```

#### 3. Execute by Task or Group
Point the CLI to both files using repeated `--custom-tasks-file` flags. The CLI automatically loads Python code first, so YAML references resolve perfectly:

```bash
# Run a specific leaf task
ai-edge-eval \
      --runner litert-lm --model path/to/model.litertlm \
      --framework custom \
      --custom-tasks-file plugins.py \
      --custom-tasks-file tasks.yaml \
      --tasks my_project:qa:dev

# Run an entire task group (automatically expands to all matching subtasks)
ai-edge-eval \
      --runner litert-lm --model path/to/model.litertlm \
      --framework custom \
      --custom-tasks-file plugins.py \
      --custom-tasks-file tasks.yaml \
      --tasks my_project:qa
```

---

### Approach 2: Pure Python Scripting

For quick one-off tasks, you can author a self-contained `CustomTask` object and register it directly:

```python
# File: register_oneoff.py

from model_eval.custom_tasks import CustomTask, DatasetRow, TaskRegistry
from model_eval.config.generation_config import GenerationConfig

qa_task = CustomTask(
    name="simple_qa",
    dataset=lambda: iter([DatasetRow(messages=[{"role": "user", "content": "2+2"}], ground_truth="4")]),
    metric_fn=lambda p, g, r: {"accuracy": 1.0 if p[0].strip() == g[0] else 0.0},
    generation_config=GenerationConfig(temperature=0.0, max_new_tokens=20)
)
TaskRegistry.global_registry().register(qa_task)
```

```bash
ai-edge-eval \
      --runner litert-lm --model path/to/model.litertlm \
      --framework custom \
      --custom-tasks-file register_oneoff.py \
      --tasks simple_qa
```

---

## 🔍 Discovery Commands

`ai-edge-eval` includes built-in discovery utilities to help you explore supported configurations, tasks, and runners.

### Argument Discovery
Use the `list-args` subcommand to inspect available configurations and parameters exposed by a given runner or evaluation framework:

```bash
# Discover runner arguments
ai-edge-eval list-args --runner litert-lm

# Discover evaluation framework arguments
ai-edge-eval list-args --framework lm-eval
```

### Custom Framework Discovery
Use the `list-metrics` and `list-loaders` subcommands to inspect reusable extension machinery registered in your custom Python scripts:

```bash
ai-edge-eval list-metrics --custom-tasks-file plugins.py
ai-edge-eval list-loaders --custom-tasks-file plugins.py
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
