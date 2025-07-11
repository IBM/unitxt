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
        "dialog": [
            {
                "role": "user",
                "content": "What is the sum of  23098082, 32709872 and 2329398? Use tool call for each math operation",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "tool_call_0",
                        "function": {
                            "name": "add_numbers",
                            "arguments": {"num1": 23098082, "num2": 32709872},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "add_numbers",
                "tool_call_id": "tool_call_0",
                "content": "55807954",
            },
        ],
        "tools": [sum_tool, subtract_tool],
        "reference_calls": [
            {
                "name": "add_numbers",
                "arguments": {"num1": 55807954, "num2": 2329398},
            },
            {
                "name": "add_numbers",
                "arguments": {"num1": 2329398, "num2": 55807954},
            },
        ],
    },
    {
        "dialog": [
            {
                "role": "user",
                "content": "Subtract  1212382312231231 from 834672468234768234?",
            },
        ],
        "tools": [sum_tool, subtract_tool],
        "reference_calls": [
            {
                "name": "subtract_numbers",
                "arguments": {"num1": 834672468234768234, "num2": 1212382312231231},
            }
        ],
    },
]

dataset = create_dataset(
    task="tasks.tool_calling.multi_turn",
    test_set=data,
    split="test",
    format="formats.chat_api",
    max_test_instances=10,
)

model = CrossProviderInferenceEngine(
    model="granite-3-3-8b-instruct", provider="watsonx", temperature=0
)


predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Instance Results:")
print(results.instance_scores)

print("Global Results:")
print(results.global_scores.summary)
