from typing import Any, List

from unitxt import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.llm_as_judge import CreateCriteriaFromString
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import NullTemplate

data = {
    "test": [
        {
            "question": "How is the weather?",
            "judgement": "The temperature is described in both Fahrenheit and Celsius.",
        },
        {
            "question": "Tell me a joke about cats",
            "judgement": "Is the response funny?",
        },
    ]
}

card = TaskCard(
    loader=LoadFromDictionary(data=data, data_classification_policy=["public"]),
    preprocess_steps=[
        CreateCriteriaFromString(field="judgement", to_field="criteria"),
    ],
    task=Task(
        input_fields={"question": str},
        reference_fields={"criteria": Any},
        prediction_type=List[str],
        metrics=[
            "metrics.llm_as_judge.pairwise.rits.llama3_1_70b[context_fields=question,criteria_field=criteria]"
        ],
        default_template=NullTemplate(),
    ),
)

test_dataset = load_dataset(card=card, split="test")

predictions = [
    [
        """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
        """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
        """On most days, the weather is warm and humid. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    ],
    [
        """Why did the cat cross the road? To cat to the other side.""",
        """Why did the cat sit on the computer? Because it wanted to keep an eye on the mouse!""",
        """What is red, yellow and green? A traffic light.""",
    ],
]

results = evaluate(predictions=predictions, data=test_dataset)

print("Global Scores:")
print(results.global_scores.summary)

print("Instance Scores:")
print(results.instance_scores.summary)
