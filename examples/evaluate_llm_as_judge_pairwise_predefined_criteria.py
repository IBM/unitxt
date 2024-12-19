from typing import Any, List

from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.llm_as_judge_operators import LoadCriteria
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import NullTemplate
from unitxt.text_utils import print_dict

logger = get_logger()
data = {
    "test": [
        {
            "question": "How is the weather?",
            "criteria": "metrics.llm_as_judge.pairwise.criterias.temperature_in_celsius_and_fahrentheit",
        },
        {
            "question": "Tell me a joke about cats",
            "criteria": "metrics.llm_as_judge.pairwise.criterias.funny_joke",
        },
    ]
}

card = TaskCard(
    loader=LoadFromDictionary(data=data, data_classification_policy=["public"]),
    preprocess_steps=[
        LoadCriteria(field="criteria", to_field="criteria"),
    ],
    task=Task(
        input_fields={"question": str},
        reference_fields={"criteria": Any},
        prediction_type=List[str],
        metrics=[
            "metrics.llm_as_judge.pairwise.watsonx.llama3_1_70b[context_fields=question,criteria_field=criteria]"
        ],
        default_template=NullTemplate(),
    ),
)

test_dataset = load_dataset(card=card)["test"]

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

evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
        ],
    )
