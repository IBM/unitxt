from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.inference import (
    HFAutoModelInferenceEngine,
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
        "query": "What is the sum of  1212382312231231 and 834672468234768234?",
        "tools": [sum_tool, subtract_tool],
        "reference_calls": [
            {
                "name": "add_numbers",
                "arguments": {"num1": 1212382312231231, "num2": 834672468234768234},
            },
            {
                "name": "add_numbers",
                "arguments": {"num1": 834672468234768234, "num2": 1212382312231231},
            },
        ],
    },
    {
        "query": "Subtract  1212382312231231 from 834672468234768234?",
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
    task="tasks.tool_calling.supervised",
    test_set=data,
    split="test",
    format="formats.chat_api",
    max_test_instances=10,
)

model = HFAutoModelInferenceEngine(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens=512,
    batch_size=1,
)


predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Instance Results:")
print(results.instance_scores)

print("Global Results:")
print(results.global_scores.summary)
