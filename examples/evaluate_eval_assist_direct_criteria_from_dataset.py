import json
from typing import Any

from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.artifact import fetch_artifact
from unitxt.blocks import Task, TaskCard
from unitxt.eval_assist_constants import CriteriaOption, CriteriaWithOptions
from unitxt.loaders import LoadFromDictionary
from unitxt.operators import FieldOperator
from unitxt.text_utils import print_dict

logger = get_logger()


class LoadCriteria(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return fetch_artifact(text)[0]


class CreateCriteriaFromDict(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return CriteriaWithOptions(
            name=text["name"],
            description=text["description"],
            options=[
                CriteriaOption(
                    name=option_dict["name"],
                    description=option_dict["description"],
                )
                for option_dict in text["options"]
            ],
        )


class CreateCriteriaFromJson(CreateCriteriaFromDict):
    def process_value(self, text: Any) -> Any:
        dict = json.loads(text)
        return super().process_value(dict)


class CreateCriteriaFromString(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return CriteriaWithOptions(
            name=f"Unknown ({text[:20]}...)",
            description=text,
            options=[
                CriteriaOption(name="Yes", description=""),
                CriteriaOption(name="No", description=""),
            ],
            option_map={
                "Yes": 1.0,
                "No": 0.0,
            },
        )


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
        CreateCriteriaFromString(field="judgement", to_field="criteria"),
    ],
    task=Task(
        input_fields={"question": str, "criteria": Any},
        reference_fields={},
        prediction_type=str,
        metrics=[
            "metrics.llm_as_judge.eval_assist.direct_assessment.rits.llama3_1_70b[context_fields=question]"
        ],
    ),
)

test_dataset = load_dataset(card=card, template="templates.empty")["test"]

predictions = [
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
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
