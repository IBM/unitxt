from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import InputOutputTemplate
from unitxt.text_utils import print_dict

logger = get_logger()

# Set up question answer pairs in a dictionary
data = {
    "test": [
        {"question": "What is the capital of Texas?", "answer": "Austin"},
        {"question": "What is the color of the sky?", "answer": "Blue"},
    ]
}

card = TaskCard(
    # Load the data from the dictionary.  Data can be  also loaded from HF, CSV files, COS and other sources
    # with different loaders.
    loader=LoadFromDictionary(data=data),
    # Define the QA task input and output and metrics.
    task=Task(
        input_fields={"question": str},
        reference_fields={"answer": str},
        prediction_type=str,
        metrics=["metrics.accuracy"],
    ),
)

# Create a simple template that formats the input.
# Add lowercase normalization as a post processor.

template = InputOutputTemplate(
    instruction="Answer the following question.",
    input_format="{question}",
    output_format="{answer}",
    postprocessors=["processors.lower_case"],
)
# Verbalize the dataset using the template
dataset = load_dataset(
    card=card,
    template=template,
    format="formats.chat_api",
    split="test",
    max_test_instances=10,
)


# Infer using Llama-3.2-1B base using HF API
engine = HFPipelineBasedInferenceEngine(
    model_name="meta-llama/Llama-3.2-1B", max_new_tokens=32
)
# Change to this to infer with external APIs:
# CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]


predictions = engine.infer(dataset)
evaluated_dataset = evaluate(predictions=predictions, data=dataset)

# Print results
for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
        ],
    )
