from typing import List

from unitxt import evaluate
from unitxt.api import create_dataset
from unitxt.blocks import Task
from unitxt.templates import NullTemplate

print("Answer relevance evaluation")

data = [
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports a variety of foundation models",
        "contexts": ["Supported foundation models available with watsonx.ai"],
    }
]

dataset = create_dataset(
    task=Task(
        input_fields={"contexts": List[str], "answer": str, "question": str},
        reference_fields={},
        prediction_type=float,
        default_template=NullTemplate(),
        metrics=["metrics.granite_guardian.rag_risk.answer_relevance"],
    ),
    test_set=data,
    split="test",
)

results = evaluate(predictions=[0.0], data=dataset)

print("Global Scores:")
print(results.global_scores.summary)

print("Instance Scores:")
print(results.instance_scores.summary)
