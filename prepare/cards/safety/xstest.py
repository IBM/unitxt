from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    Task,
    TaskCard,
)
from unitxt.operators import Set
from unitxt.splitters import RenameSplits
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="Paul/XSTest", data_classification_policy=["public"]
    ),
    preprocess_steps=[
        RenameSplits({"train": "test"}),
        Set({"response": ""}),
    ],
    task=Task(
        input_fields={"prompt": str, "id": int, "type": str, "label": str},
        reference_fields={"response": str},
        prediction_type=str,
        metrics=[
            "metrics.llm_as_judge.direct.watsonx.llama3_3_70b[criteria=metrics.llm_as_judge.direct.criteria.assistant_message_compliance, context_fields=[prompt,response]]"
        ],
    ),
    templates={"default": InputOutputTemplate(input_format="{prompt}", output_format="{response}")},
    __description__="Test suite designed to identify exaggerated safety / false refusal in Large Language Models (LLMs).",
    __tags__={
        "languages": ["english"],
    },
)

test_card(card, strict=False, demos_taken_from="test", num_demos=0)

add_to_catalog(card, "cards.safety.xstest", overwrite=True)
