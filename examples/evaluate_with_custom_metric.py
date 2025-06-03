from typing import Dict, List

from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.blocks import Task
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.metrics import InstanceMetric
from unitxt.templates import InputOutputTemplate

logger = get_logger()

# Set up question answer pairs in a dictionary
data = [
    {"text": "John paid Apple $100 dollars."},
    {"text": "IBM was paid 200 dollars by Phil"},
]


class IsValidJson(InstanceMetric):
    main_score = "valid_json"  # name of the main score
    reduction_map = {
        "mean": ["valid_json"]
    }  # defines that the global score is a mean of the instance scores
    ci_scores = [
        "valid_json"
    ]  # define that confidence internal should be calculated on the score
    prediction_type = str  # the metric expect the prediction as an int

    def compute(
        self, references: List[str], prediction: str, task_data: List[Dict]
    ) -> dict:
        try:
            import json

            json.loads(prediction)
            return {
                self.main_score: 1.0,
                "error": "no errors. successfully parsed json.",
            }
        except Exception as e:
            return {self.main_score: 0, "error": str(e)}


# define the QA task
task = Task(
    input_fields={"text": str},
    reference_fields={},
    prediction_type=str,
    metrics=[IsValidJson()],
)


# Create a simple template that formats the input.
# Add lowercase normalization as a post processor.

template = InputOutputTemplate(
    instruction="Extract the company name and amount as a json with two keys COMPANY_NAME and AMOUNT. Return only the a valid json that can be parsed, without any explanations or prefixes and suffixes",
    input_format="{text}",
    output_format="",
)
# Verbalize the dataset using the template
dataset = create_dataset(
    task=task, test_set=data, template=template, format="formats.chat_api", split="test"
)


# Infer using SmolLM2 using HF API
# model = HFPipelineBasedInferenceEngine(
#    model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_new_tokens=32
# )
# Change to this to infer with external APIs:
# from unitxt.inference import CrossProviderInferenceEngine
model = CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam". "rits"]


predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Instance Results:")
print(results.instance_scores)

print("Global Results:")
print(results.global_scores.summary)
