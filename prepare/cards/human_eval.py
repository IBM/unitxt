from unitxt import add_to_catalog
from unitxt.blocks import (
    FormTask,
    InputOutputTemplate,
    LoadHF,
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
    __tags__={
        "dataset_info_tags": [
            "task_categories:text2text-generation",
            "annotations_creators:expert-generated",
            "language_creators:expert-generated",
            "multilinguality:monolingual",
            "size_categories:n<1K",
            "source_datasets:original",
            "language:en",
            "license:mit",
            "code-generation",
            "croissant",
            "arxiv:2107.03374",
            "region:us",
        ]
    },
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
