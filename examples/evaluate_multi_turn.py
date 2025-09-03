from typing import List

from unitxt.api import create_dataset, evaluate
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.task import Task
from unitxt.templates import MultiTurnTemplate
from unitxt.types import Conversation

data = [
    {
        "conversation": {
            "id": "1",
            "dialog": [
                {
                    "role": "system",
                    "content": "Have a dialog with the user and answer the questions.",
                },
                {"role": "user", "content": "Where is Paris?"},
            ],
        },
        "answers": ["Paris is in France"],
    },
    {
        "conversation": {
            "id": "1",
            "dialog": [
                {
                    "role": "system",
                    "content": "Have a dialog with the user and answer the questions.",
                },
                {"role": "user", "content": "Where is Paris?"},
                {"role": "assistant", "content": "Paris is in France"},
                {"role": "user", "content": "How is it also called?"},
            ],
        },
        "answers": ["The City of Lights"],
    },
    {
        "conversation": {
            "id": "2",
            "dialog": [
                {"role": "system", "content": "Calculate and return only the number."},
                {"role": "user", "content": "12+13"},
            ],
        },
        "answers": ["25"],
    },
    {
        "conversation": {
            "id": "2",
            "dialog": [
                {"role": "system", "content": "Calculate and return only the number."},
                {"role": "user", "content": "12+13"},
                {"role": "assistant", "content": "25"},
                {"role": "system", "content": "Multiply the result by 3.14159"},
            ],
        },
        "answers": ["78.53975"],
    },
    {
        "conversation": {
            "id": "2",
            "dialog": [
                {"role": "system", "content": "Calculate and return only the number."},
                {"role": "user", "content": "12+13"},
                {"role": "assistant", "content": "25"},
                {"role": "system", "content": "Multiply the result by 3.14159"},
                {"role": "assistant", "content": "25"},
                {"role": "system", "content": "Multiply the result by 0"},
            ],
        },
        "answers": ["0"],
    },
]


template = MultiTurnTemplate(
    references_field="answers",
    turns_field="conversation/dialog",
)


criterion = "metrics.llm_as_judge.direct.criteria.correctness_based_on_ground_truth"
llm_as_judge_metric = f"metrics.llm_as_judge.direct.watsonx.llama3_3_70b[criteria={criterion}, context_fields=[answers]]"


task = Task(
    input_fields={
        "conversation": Conversation,
    },
    reference_fields={
        "answers": List[str],
    },
    prediction_type=str,
    metrics=[
        llm_as_judge_metric,
        "metrics.rouge",
        "metrics.normalized_sacrebleu",
        "metrics.accuracy",
    ],
    default_template=template,
)
model = CrossProviderInferenceEngine(model="llama-3-3-70b-instruct", provider="watsonx")

dataset = create_dataset(
    task=task,
    test_set=data,
    format="formats.chat_api",
    split="test",
    max_test_instances=10,
)
predictions = model.infer(dataset)

results = evaluate(predictions=predictions, data=dataset)

print("Instance Results:")
print(results.instance_scores.summary)

print("Global Results:")
print(results.global_scores.summary)
