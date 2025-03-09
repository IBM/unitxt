<div align="center">
    <img src="https://raw.githubusercontent.com/IBM/unitxt/main/assets/banner.png" alt="Image Description" width="100%" />
</div>

[![Button](https://img.shields.io/badge/Video-pink?style=for-the-badge)](https://unitxt.readthedocs.io/en/latest/_static/video.mov)
[![Button](https://img.shields.io/badge/Documentation-pink?style=for-the-badge)](https://unitxt.readthedocs.io/en/latest/docs/introduction.html)
[![Button](https://img.shields.io/badge/Demo-pink?style=for-the-badge)](https://unitxt.readthedocs.io/en/latest/docs/demo.html)
[![Button](https://img.shields.io/badge/Tutorial-pink?style=for-the-badge)](https://unitxt.readthedocs.io/en/latest/docs/adding_dataset.html)
[![Button](https://img.shields.io/badge/Paper-pink?style=for-the-badge)](https://arxiv.org/abs/2401.14019)
[![Button](https://img.shields.io/badge/Catalog-pink?style=for-the-badge)](https://unitxt.readthedocs.io/en/latest/catalog/catalog.__dir__.html)
[![Button](https://img.shields.io/badge/Contributors-pink?style=for-the-badge)](https://github.com/IBM/unitxt/blob/main/CONTRIBUTING.md)
[![Button](https://img.shields.io/badge/PyPi-pink?style=for-the-badge)](https://pypi.org/project/unitxt/)


In the dynamic landscape of generative NLP, traditional text processing pipelines limit research flexibility and reproducibility, as they are tailored to specific dataset, task, and model combinations. The escalating complexity, involving system prompts, model-specific formats, instructions, and more, calls for a shift to a structured, modular, and customizable solution.

 Addressing this need, we present Unitxt, an innovative library for customizable textual data preparation and evaluation tailored to generative language models. Unitxt natively integrates with common libraries like HuggingFace and LM-eval-harness and deconstructs processing flows into modular components, enabling easy customization and sharing between practitioners. These components encompass model-specific formats, task prompts, and many other comprehensive dataset processing definitions. The Unitxt-Catalog centralizes these components, fostering collaboration and exploration in modern textual data workflows. Beyond being a tool, Unitxt is a community-driven platform, empowering users to build, share, and advance their pipelines collaboratively.

#
[![version](https://img.shields.io/pypi/v/unitxt)](https://pypi.org/project/unitxt/)
![license](https://img.shields.io/github/license/ibm/unitxt)
![python](https://img.shields.io/badge/python-3.8%20|%203.9-blue)
![tests](https://img.shields.io/github/actions/workflow/status/ibm/unitxt/library_tests.yml?branch=main&label=tests)
[![Coverage Status](https://coveralls.io/repos/github/IBM/unitxt/badge.svg)](https://coveralls.io/github/IBM/unitxt)
![Read the Docs](https://img.shields.io/readthedocs/unitxt)
[![downloads](https://static.pepy.tech/personalized-badge/unitxt?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/unitxt)

#

https://github.com/IBM/unitxt/assets/23455264/baef9131-39d4-4164-90b2-05da52919fdf

### 🦄 Currently on Unitxt Catalog

![Abstract Tasks](https://img.shields.io/badge/Abstract_Tasks-64-blue)
![Dataset Cards](https://img.shields.io/badge/Dataset_Cards-3174-blue)
![Templates](https://img.shields.io/badge/Templates-342-blue)
![Benchmarks](https://img.shields.io/badge/Benchmarks-6-blue)
![Metrics](https://img.shields.io/badge/Metrics-462-blue)

### 🦄 Run Unitxt Exploration Dashboard

To launch unitxt graphical user interface first install unitxt with ui requirements:
```
pip install unitxt[ui]
```
Then launch the ui by running:
```
unitxt-explore
```

# 🦄 Example

This is a simple example of running end-to-end evaluation in self contained python code over user data.

See more examples in examples subdirectory.

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

# 🦄 Contributors

Please install Unitxt from source by:
```bash
git clone git@github.com:IBM/unitxt.git
cd unitxt
pip install -e ".[dev]"
pre-commit install
```

# 🦄 Citation

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
    abstract = "In the dynamic landscape of generative NLP, traditional text processing pipelines limit research flexibility and reproducibility, as they are tailored to specific dataset, task, and model combinations. The escalating complexity, involving system prompts, model-specific formats, instructions, and more, calls for a shift to a structured, modular, and customizable solution.Addressing this need, we present Unitxt, an innovative library for customizable textual data preparation and evaluation tailored to generative language models. Unitxt natively integrates with common libraries like HuggingFace and LM-eval-harness and deconstructs processing flows into modular components, enabling easy customization and sharing between practitioners. These components encompass model-specific formats, task prompts, and many other comprehensive dataset processing definitions. The Unitxt Catalog centralizes these components, fostering collaboration and exploration in modern textual data workflows. Beyond being a tool, Unitxt is a community-driven platform, empowering users to build, share, and advance their pipelines collaboratively. Join the Unitxt community at https://github.com/IBM/unitxt",
}
```

Unitxt emoji designed by [OpenMoji](https://openmoji.org/#) - the open-source emoji and icon project. License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/#)
