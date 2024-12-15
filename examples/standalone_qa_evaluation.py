from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.blocks import Task
from unitxt.inference import HFPipelineBasedInferenceEngine
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


# Infer using Llama-3.2-1B base using HF API
engine = HFPipelineBasedInferenceEngine(
    model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
)
# Change to this to infer with external APIs:
# from unitxt.inference import CrossProviderInferenceEngine
# engine = CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam". "rits"]


predictions = engine.infer(dataset)
results = evaluate(predictions=predictions, data=dataset)

# Print results
print(
    results.instance_scores.to_df(
        columns=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
            "score_name",
        ],
    )
)
