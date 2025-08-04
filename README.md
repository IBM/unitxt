<div align="center">
    <img src="https://www.unitxt.ai/en/latest/_static/banner.png" alt="Image Description" width="100%" />
</div>

#
[![version](https://img.shields.io/pypi/v/unitxt)](https://pypi.org/project/unitxt/)
![license](https://img.shields.io/github/license/ibm/unitxt)
![python](https://img.shields.io/badge/python-3.8%20|%203.9-blue)
![tests](https://img.shields.io/github/actions/workflow/status/ibm/unitxt/library_tests.yml?branch=main&label=tests)
[![Coverage Status](https://coveralls.io/repos/github/IBM/unitxt/badge.svg)](https://coveralls.io/github/IBM/unitxt)
![Read the Docs](https://img.shields.io/readthedocs/unitxt)
[![downloads](https://static.pepy.tech/personalized-badge/unitxt?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/unitxt)

### 🦄 Unitxt is a Python library for enterprise-grade evaluation of AI performance, offering the world's largest catalog of tools and data for end-to-end AI benchmarking

#

## Why Unitxt?

- 🌐 **Comprehensive**: Evaluate text, tables, vision, speech, and code in one unified framework
- 💼 **Enterprise-Ready**: Battle-tested components with extensive catalog of benchmarks
- 🧠 **Model Agnostic**: Works with HuggingFace, OpenAI, WatsonX, and custom models
- 🔒 **Reproducible**: Shareable, modular components ensure consistent results

## Quick Links
- 📖 [Documentation](https://www.unitxt.ai)
- 🚀 [Getting Started](https://www.unitxt.ai)
- 📁 [Browse Catalog](https://www.unitxt.ai/en/latest/catalog/catalog.__dir__.html)

# Installation

```bash
pip install unitxt
```

# Quick Start

## Command Line Evaluation
```bash
# Simple evaluation
unitxt-evaluate \
    --tasks "card=cards.mmlu_pro.engineering" \
    --model cross_provider \
    --model_args "model_name=llama-3-1-8b-instruct" \
    --limit 10

# Multi-task evaluation
unitxt-evaluate \
    --tasks "card=cards.text2sql.bird+card=cards.mmlu_pro.engineering" \
    --model cross_provider \
    --model_args "model_name=llama-3-1-8b-instruct,max_tokens=256" \
    --split test \
    --limit 10 \
    --output_path ./results/evaluate_cli \
    --log_samples \
    --apply_chat_template

# Benchmark evaluation
unitxt-evaluate \
    --tasks "benchmarks.tool_calling" \
    --model cross_provider \
    --model_args "model_name=llama-3-1-8b-instruct,max_tokens=256" \
    --split test \
    --limit 10 \
    --output_path ./results/evaluate_cli \
    --log_samples \
    --apply_chat_template
```

## Loading as Dataset
Load thousands of datasets in chat API format, ready for any model:
```python
from unitxt import load_dataset

dataset = load_dataset(
    card="cards.gpqa.diamond",
    split="test",
    format="formats.chat_api",
)
```

## 📊 Available on The Catalog

![Tasks](https://img.shields.io/badge/Tasks-68-blue)
![Datasets](https://img.shields.io/badge/Datasets-3254-blue)
![Prompts](https://img.shields.io/badge/Prompts-357-blue)
![Benchmarks](https://img.shields.io/badge/Benchmarks-11-blue)
![Metrics](https://img.shields.io/badge/Metrics-584-blue)

## 🚀 Interactive Dashboard

Launch the graphical user interface to explore datasets and benchmarks:
```
pip install unitxt[ui]
unitxt-explore
```

# Complete Python Example

Evaluate your own data with any model:

```python
# Import required components
from unitxt import evaluate, create_dataset
from unitxt.blocks import Task, InputOutputTemplate
from unitxt.inference import HFAutoModelInferenceEngine

# Question-answer dataset
data = [
    {"question": "What is the capital of Texas?", "answer": "Austin"},
    {"question": "What is the color of the sky?", "answer": "Blue"},
]

# Define the task and evaluation metric
task = Task(
    input_fields={"question": str},
    reference_fields={"answer": str},
    prediction_type=str,
    metrics=["metrics.accuracy"],
)

# Create a template to format inputs and outputs
template = InputOutputTemplate(
    instruction="Answer the following question.",
    input_format="{question}",
    output_format="{answer}",
    postprocessors=["processors.lower_case"],
)

# Prepare the dataset
dataset = create_dataset(
    task=task,
    template=template,
    format="formats.chat_api",
    test_set=data,
    split="test",
)

# Set up the model (supports Hugging Face, WatsonX, OpenAI, etc.)
model = HFAutoModelInferenceEngine(
    model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
)

# Generate predictions and evaluate
predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

# Print results
print("Global Results:\n", results.global_scores.summary)
print("Instance Results:\n", results.instance_scores.summary)
```

# Contributing

Read the [contributing guide](./CONTRIBUTING.md) for details on how to contribute to Unitxt.

#

# Citation

If you use Unitxt in your research, please cite our paper:

```bib
@inproceedings{bandel-etal-2024-unitxt,
    title = "Unitxt: Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative {AI}",
    author = "Bandel, Elron  and
      Perlitz, Yotam  and
      Venezian, Elad  and
      Friedman, Roni  and
      Arviv, Ofir  and
      Orbach, Matan  and
      Don-Yehiya, Shachar  and
      Sheinwald, Dafna  and
      Gera, Ariel  and
      Choshen, Leshem  and
      Shmueli-Scheuer, Michal  and
      Katz, Yoav",
    editor = "Chang, Kai-Wei  and
      Lee, Annie  and
      Rajani, Nazneen",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: System Demonstrations)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-demo.21",
    pages = "207--215",
}
```