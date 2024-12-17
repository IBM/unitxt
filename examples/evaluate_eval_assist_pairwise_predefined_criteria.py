from typing import List

from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.text_utils import print_dict

logger = get_logger()
data = {
    "test": [
        {
            "question": "How is the weather?",
        },
        {
            "question": "Tell me a joke about cats",
        },
    ]
}

criteria = "metrics.llm_as_judge.eval_assist.pairwise_comparison.criterias.temperature_in_celsius_and_fahrentheit"
card = TaskCard(
    loader=LoadFromDictionary(data=data, data_classification_policy=["public"]),
    task=Task(
        input_fields={"question": str},
        reference_fields={},
        prediction_type=List[str],
        metrics=[
            "metrics.llm_as_judge.eval_assist.pairwise_comparison.watsonx.llama3_1_70b"
            f"[criteria={criteria}, context_fields=[question]]"
        ],
    ),
)

test_dataset = load_dataset(card=card, template="templates.empty")["test"]

predictions = [
    [
        """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
        """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
        """On most days, the weather is warm and humid. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    ],
    [
        """Why did the cat cross the road? To cat to the other side.""",
        """Why did the cat sit on the computer? Because it wanted to keep an eye on the mouse!""",
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
