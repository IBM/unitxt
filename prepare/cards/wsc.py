from src.unitxt.blocks import InputOutputTemplate, LoadHF, TemplatesList
from src.unitxt.card import TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.prepare_utils.card_types import addClassificationChoices
from src.unitxt.task import FormTask
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="super_glue", name="wsc"),
    preprocess_steps=[
        "splitters.small_no_test",
        *addClassificationChoices("label", {"0": "False", "1": "True"}),
    ],
    task=FormTask(
        inputs=["choices", "text", "span1_text", "span2_text"],
        outputs=["label"],
        metrics=["metrics.accuracy"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="""
                    Given this sentence: {text} classify if "{span2_text}" refers to "{span1_text}".
                """.strip(),
                output_format="{label}",
            ),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.wsc", overwrite=True)
