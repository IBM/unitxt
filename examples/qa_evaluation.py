from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import TaskCard
from unitxt.collections_operators import Wrap
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.loaders import LoadFromDictionary
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
    # Use the standard open qa QA task input and output and metrics.
    # It has "question" input field and "answers" output field.
    # The default evaluation metric used is rouge.
    task="tasks.qa.open",
    # Because the standand QA tasks supports multiple references in the "answers" field,
    # we wrap the raw dataset's "answer" field in a list and store in a the "answers" field.
    preprocess_steps=[Wrap(field="answer", inside="list", to_field="answers")],
)

# Verbalize the dataset using the catalog template which adds an instructio "Answer the question.",
# and "Question:"/"Answer:" prefixes.
#
# "Answer the question.
#  Question:
#  What is the color of the sky?
#  Answer:
# "
dataset = load_dataset(
    card=card,
    template="templates.qa.open.title",
    format="formats.chat_api",
    split="test",
    max_test_instances=5,
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
