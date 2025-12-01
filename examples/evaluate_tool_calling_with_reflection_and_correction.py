import json

from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

logger = get_logger()

sum_tool = {
    "name": "add_numbers",
    "description": "Add two numbers together and return the result.",
    "parameters": {
        "type": "object",
        "properties": {
            "num1": {"type": "number", "description": "The first number to add."},
            "num2": {"type": "number", "description": "The second number to add."},
        },
        "required": ["num1", "num2"],
    },
}

subtract_tool = {
    "name": "subtract_numbers",
    "description": "Subtracts one number from another from the other.",
    "parameters": {
        "type": "object",
        "properties": {
            "num1": {"type": "number", "description": "Number to subtract from"},
            "num2": {"type": "number", "description": "Number to subtract"},
        },
        "required": ["num1", "num2"],
    },
}

data = [
    {
        "query": "What is the sum of  1212382 and 834672?",
        "tools": [sum_tool, subtract_tool],
        "reference_calls": [
            {
                "name": "add_numbers",
                "arguments": {"num1": 1212382, "num2": 834672},
            },
            {
                "name": "add_numbers",
                "arguments": {"num1": 834672, "num2": 1212382},
            },
        ],
    },
    {
        "query": "Subtract  12123  from 83467",
        "tools": [sum_tool, subtract_tool],
        "reference_calls": [
            {
                "name": "subtract_numbers",
                "arguments": {"num1": 83467, "num2": 12123},
            }
        ],
    },
]

dataset = create_dataset(
    task="tasks.tool_calling.supervised",
    test_set=data,
    split="test",
    format="formats.chat_api",
    metrics=[
        "metrics.tool_calling.reflection",
    ],
    max_test_instances=10,
)

model = CrossProviderInferenceEngine(
    model="granite-3-3-8b-instruct", provider="watsonx", temperature=0
)


predictions = model(dataset)
# Insert errors ito the model predictions to check reflector
predictions[0] = predictions[0].replace("8", "7")
predictions[0] = predictions[0].replace("num1", "num3")
predictions[1] = predictions[1].replace("subtract_numbers", "multiply_numbers")

results = evaluate(predictions=predictions, data=dataset)
print("Instance Results:")
print(results.instance_scores.summary)

print("Global Results:")
print(results.global_scores.summary)


def find_elements(obj, element_name):
    results = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == element_name:
                results.append(value)
            else:
                results.extend(find_elements(value, element_name))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(find_elements(item, element_name))
    return results


# Collect the corrected tool calls when exists


corrected_predictions = []
for instance_scores, prediction in zip(results.instance_scores, predictions):
    raw_responses = find_elements(instance_scores, "raw_response")
    corrected_tool_calls = find_elements(raw_responses, "tool_call")
    if len(corrected_tool_calls) > 0:
        corrected_predictions.append(json.dumps(corrected_tool_calls[0]))
    else:
        corrected_predictions.append(prediction)

# Run again on the original dataset and metric
dataset = create_dataset(
    task="tasks.tool_calling.supervised",
    test_set=data,
    split="test",
    format="formats.chat_api",
    metrics=["metrics.tool_calling.key_value.accuracy"],
    max_test_instances=10,
)

corrected_results = evaluate(predictions=corrected_predictions, data=dataset)

print("Instance Results (After reflection):")
print(corrected_results.instance_scores.summary)

print("Global Results (After reflection):")
print(corrected_results.global_scores.summary)
