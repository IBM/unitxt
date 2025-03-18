from unitxt import get_logger
from unitxt.api import create_dataset, evaluate

logger = get_logger()

# Set up question answer pairs in a dictionary
dataset = [
    {"question": "What is the capital of Texas?", "answers": ["Austin"]},
    {"question": "What is the color of the sky?", "answers": ["Blue"]},
]

predictions = ["San Antonio", "blue"]

dataset = create_dataset(
    task="tasks.qa.open",
    test_set=dataset,
    metrics=[
        "metrics.qa.open.recommended_no_gpu",
        # "metrics.qa.open.recommended_llm_as_judge",
    ],
)

results = evaluate(predictions, dataset["test"])

print("Global Results:")
print(results.global_scores.summary)

print("Instance Results:")
print(results.instance_scores.summary)
