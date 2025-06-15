import json

from unitxt import get_logger, load_dataset
from unitxt.api import LoadFromDictionary, TaskCard, evaluate
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate

logger = get_logger()

# Set up question answer pairs in a dictionary
data = [
    {
        "input": "What is the capital of Texas?",
        "output": "The capital of Texas is Austin",
    },
    {
        "input": "Count until three.",
        "output": "1, 2, 3.",
    },
    {
        "input": "Tell me a joke about chicken crossing the road.",
        "output": "Why did the checken cross the road? To get to the other side.",
    },
]


# Create a unitxt cards that converts the input data to the format required by the
# t`asks.qa.multiple_choice.open task`.
#
# It concatenates the different options fields to the 'choices' field.
# And sets the 'answer' field, to the index of the correct answer in the 'choices' field.
card = TaskCard(
    loader=LoadFromDictionary(data={"test": data}),
    task=Task(
        input_fields={
            "input": "str",
        },
        reference_fields={
            "output": "str",
        },
        prediction_type="str",
        metrics=[
            "metrics.rouge",
        ],
        default_template=InputOutputTemplate(
            input_format="{input}",
            output_format="{output}",
        ),
    ),
)

dataset = load_dataset(
    card=card,
    split="test",
    format="formats.chat_api",
)

# Change to this to infer with external APIs:
model = CrossProviderInferenceEngine(
    model="ibm/granite-3-2-8b-instruct", provider="watsonx-sdk"
)
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]

predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Example prompt:")
print(json.dumps(results.instance_scores[0]["source"], indent=4))


print("Instance Results:")
print(results.instance_scores)

print("Global Results:")
print(results.global_scores.summary)

print("Print Yaml Representation of DataSet TaskCard")
print(card.to_yaml())
