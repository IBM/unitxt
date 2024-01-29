from unitxt import add_to_catalog
from unitxt.blocks import (
    FormTask,
    InputOutputTemplate,
    LoadHF,
    TaskCard,
    TemplatesList,
)
from unitxt.operators import ExecuteQuery
from unitxt.test_utils.card import test_card

get_asserts = """
import re
import json

_ASSERT_REGEX = r"assert.*?(?=\\n\\s*assert|$)"
json.dumps(
    [
        t.replace("candidate", entry_point)
        for t in re.findall(_ASSERT_REGEX, test, re.DOTALL)
    ]
)
"""


card = TaskCard(
    loader=LoadHF(path="openai_humaneval", split="test"),
    preprocess_steps=[ExecuteQuery(query=get_asserts, to_field="test_list")],
    task=FormTask(
        inputs=["prompt"],
        outputs=["prompt", "canonical_solution", "test_list"],
        metrics=["metrics.bleu"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="{prompt}\n",
                output_format="{prompt}\n{canonical_solution}",
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
add_to_catalog(card, "cards.human_eval", overwrite=True)
