from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.blocks import Task
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.templates import InputOutputTemplate

logger = get_logger()

# Set up question answer pairs in a dictionary
data = [
    {"question": "What is the capital of Texas?", "answer": "Austin"},
    {"question": "What is the color of the sky?", "answer": "Blue"},
]


# define the QA task
task = Task(
    input_fields={"question": str},
    reference_fields={"answer": str},
    prediction_type=str,
    metrics=["metrics.rouge"],
)


# Create a simple template that formats the input.
# Add lowercase normalization as a post processor.

template = InputOutputTemplate(
    instruction="Answer the following question in one word.",
    input_format="{question}",
    output_format="{answer}",
    postprocessors=["processors.lower_case"],
)
# Verbalize the dataset using the template
dataset = create_dataset(
    task=task, test_set=data, template=template, format="formats.chat_api", split="test"
)


model = CrossProviderInferenceEngine(
    model="SmolLM2-1.7B-Instruct", provider="hf", use_cache=False
)
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "rits", "hf"]
# (model must be available in the provider service)

predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Instance Results:")
print(results.instance_scores.summary)

print("Global Results:")
print(results.global_scores.summary)
