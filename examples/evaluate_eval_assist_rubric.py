import os

from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import InputOutputTemplate, TemplatesDict
from unitxt.text_utils import print_dict
from typing import Any
from unitxt.operators import Set

logger = get_logger()

rubric_json = {
    "name": "Temperature",
    "criteria": "In the response, if there is a numerical temperature present, is it denominated in both Fahrenheit and Celsius?",
    "options": [
        {
            "option": "Yes",
            "description": "The temperature reading is provided in both Fahrenheit and Celsius."
        },
        {
            "option": "No",
            "description": "The temperature reading is provided either in Fahrenheit or Celsius, but not both."
        },
        {
            "option": "Pass",
            "description": "There is no numerical temperature reading in the response."
        }
    ]
}

data = {
    "test": [
       {"question": "How is the weather?"},
        {"question": "How is the weather?"},
        {"question": "How is the weather?"},
    ],
}

card = TaskCard(
    # Load the data from the dictionary.  Data can be  also loaded from HF, CSV files, COS and other sources
    # with different loaders.
    loader=LoadFromDictionary(data=data),
    preprocess_steps=[Set(fields={"rubric": rubric_json})],
    # Define the QA task input and output and metrics.
    task=Task(
        input_fields={"question": str, "rubric": dict[str, Any]},
        reference_fields={},
        prediction_type=str,
        metrics = ["metrics.llm_as_judge.eval_assist.direct.prometheus"],
    ),
    # Create a simple template that formats the input.
    # Add lowercase normalization as a post processor.
    templates=TemplatesDict(
        {
            "simple": InputOutputTemplate(
                instruction="Answer the following question.",
                input_format="{question}",
                output_format="",
                postprocessors=["processors.lower_case"],
            )
        }
    ),
)

test_dataset = load_dataset(card=card, template_card_index="simple")["test"]
predictions = ["""On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants."""]
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
