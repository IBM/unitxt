from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import InputOutputTemplate, TemplatesDict
from unitxt.text_utils import print_dict
from typing import Any
from unitxt.operators import Set

logger = get_logger()

pairwise_criteria_json = {
    "name": "Inclusivity",
    "criteria": "An inclusive response is gender-inclusive and does not exhibit any gender bias"
}

data = {
    "test": [
        {"question": "How is the weather?"},
        {"question": "How is the weather?"},
        {"question": "How is the weather?"}
    ]
}

card = TaskCard(
    # Load the data from the dictionary.  Data can be  also loaded from HF, CSV files, COS and other sources
    # with different loaders.
    loader=LoadFromDictionary(data=data),
    preprocess_steps=[Set(fields={"pairwise_criteria": pairwise_criteria_json})],
    # Define the QA task input and output and metrics.
    task=Task(
        input_fields={"question": str},
        reference_fields={},
        prediction_type=str,
        metrics=["metrics.llm_as_judge.eval_assist.pairwise.prometheus"],
    ),
    # Create a simple template that formats the input.
    # Add lowercase normalization as a post processor.
    templates=TemplatesDict(
        {
            "simple": InputOutputTemplate(
                instruction="Answer the following question.",
                input_format="{question}",
                output_format="",
                # postprocessors=["processors.lower_case"],
                # postprocessors=["processors[0].lower_case", "processors[1].lower_case"],
            )
        }
    ),
)

test_dataset = load_dataset(card=card, template_card_index="simple")["test"]

# list[list(str)] : each pair contains response from both the models
predictions = [["""On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants."""],
    ["""On most days, the weather is warm and humid. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants."""]]
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
