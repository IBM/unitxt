from unitxt import add_to_catalog
from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    Task,
    TaskCard,
    TemplatesList,
)
from unitxt.operators import ExecuteExpression
from unitxt.settings_utils import get_settings
from unitxt.test_utils.card import test_card

settings = get_settings()
orig_settings = settings.allow_unverified_code
settings.allow_unverified_code = True


get_asserts = '[t for t in re.findall(r"assert.*?(?=\\n\\s*assert|$)", test.replace("candidate", entry_point), re.DOTALL)]'


card = TaskCard(
    loader=LoadHF(path="openai_humaneval", split="test"),
    preprocess_steps=[
        ExecuteExpression(
            expression=get_asserts, imports_list=["re"], to_field="test_list"
        )
    ],
    task=Task(
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

settings.allow_unverified_code = orig_settings
