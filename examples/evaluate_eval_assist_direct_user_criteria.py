from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.eval_assist_constants import OptionSelectionStrategyEnum
from unitxt.loaders import LoadFromDictionary
from unitxt.operators import Set
from unitxt.text_utils import print_dict

logger = get_logger()

criteria_json = {
    "name": "Temperature",
    "description": "In the response, if there is a numerical temperature present, is it denominated in both Fahrenheit and Celsius?",
    "options": [
        {
            "name": "Yes",
            "description": "The temperature reading is provided in both Fahrenheit and Celsius.",
        },
        {
            "name": "No",
            "description": "The temperature reading is provided either in Fahrenheit or Celsius, but not both.",
        },
        {
            "name": "Pass",
            "description": "There is no numerical temperature reading in the response.",
        },
    ],
    "option_map": {"Yes": 1.0, "No": 0.5, "Pass": 0.0},
}

data = {
    "test": [
        {"question": "How is the weather?"},
    ]
}

card = TaskCard(
    loader=LoadFromDictionary(data=data, data_classification_policy=["public"]),
    preprocess_steps=[Set(fields={"criteria": criteria_json})],
    task=Task(
        input_fields={"question": str, "criteria": dict},
        reference_fields={},
        prediction_type=str,
        metrics=["metrics.llm_as_judge.eval_assist.direct_assessment.rits.llama3_1_70b"],
    ),
)

test_dataset = load_dataset(card=card, template="templates.empty")["test"]

predictions = [
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
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
