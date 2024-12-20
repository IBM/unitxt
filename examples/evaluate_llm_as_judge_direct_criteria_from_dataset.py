from typing import Any

from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.llm_as_judge_operators import CreateYesNoCriteriaFromString
from unitxt.loaders import LoadFromDictionary
from unitxt.text_utils import print_dict

data = {
    "test": [
        {
            "question": "How is the weather?",
            "judgement": "In the response, if there is a numerical temperature present, is it denominated in both Fahrenheit and Celsius?",
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
        CreateYesNoCriteriaFromString(field="judgement", to_field="criteria"),
    ],
    task=Task(
        input_fields={"question": str},
        reference_fields={"criteria": Any},
        prediction_type=str,
        metrics=[
            "metrics.llm_as_judge.direct.watsonx.llama3_1_70b[context_fields=question,criteria_field=criteria]"
        ],
    ),
)

test_dataset = load_dataset(card=card, template="templates.empty")["test"]

predictions = [
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """Why did the cat cross the road? To cat to the other side.""",
]

evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "source",
            "prediction",
            # "processed_prediction",
            # "references",
            "score",
        ],
    )
