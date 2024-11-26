from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import InputOutputTemplate, TemplatesDict
from unitxt.text_utils import print_dict
from unitxt.operators import Set

criteria_json = {
    "name": "Temperature",
    "description": "In the response, if there is a numerical temperature present, is it denominated in both Fahrenheit and Celsius?",
}

data = {
    "test": [
        {"context": {"Question": "How is the weather?"}},
        {"context": {"Question": "How is the weather?"}},
        {"context": {"Question": "How is the weather?"}},
    ]
}

card = TaskCard(
    loader=LoadFromDictionary(data=data, data_classification_policy=["public"]),
    preprocess_steps=[Set(fields={"criteria": criteria_json})],
    task=Task(
        input_fields={"context": dict, "criteria": dict},
        reference_fields={},
        prediction_type=str,
        metrics=["metrics.llm_as_judge.eval_assist.pairwise_comparison.llama3_1_70b"],
    ),
    templates=TemplatesDict(
        {
            "simple": InputOutputTemplate(
                instruction="Answer the following question.",
                input_format="{context}",
                output_format="",
            )
        }
    ),
)

test_dataset = load_dataset(card=card, template_card_index="simple")["test"]

predictions = [
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants."""
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
    