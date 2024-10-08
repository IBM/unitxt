import os

from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import InputOutputTemplate, TemplatesDict
from unitxt.text_utils import print_dict

logger = get_logger()

# os.environ["GENAI_KEY"] = ""

data = {
    "test": [
       {"question": "How is the weather?"},
        {"question": "How is the weather?"},
        {"question": "How is the weather?"},
    ],
}

rubric = "metrics.llm_as_judge.eval_assist.direct.rubrics.conciseness"
metric = f"metrics.llm_as_judge.eval_assist.direct.mixtral[rubric={rubric}]"

card = TaskCard(
    loader=LoadFromDictionary(data=data),
    task=Task(
        input_fields={"question": str},
        reference_fields={},
        prediction_type=str,
        metrics = [metric]
    ),
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
