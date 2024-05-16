import os

from unitxt import add_to_catalog
from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    Task,
    TaskCard,
    TemplatesList,
)
from unitxt.operators import JoinStr
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="mbpp", name="full", split="test"),
    preprocess_steps=[
        JoinStr(field_to_field={"test_list": "test_list_str"}, separator=os.linesep),
    ],
    task=Task(
        inputs=["text", "test_list_str"],
        outputs=["test_list", "code"],
        metrics=["metrics.bleu"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format='"""{text}\n\n{test_list_str}"""',
                output_format="{code}",
            ),
        ]
    ),
)

test_card(
    card,
    demos_taken_from="test",
    demos_pool_size=1,
    num_demos=0,
    strict=False,
    loader_limit=500,
    debug=False,
)
add_to_catalog(card, "cards.mbpp", overwrite=True)
