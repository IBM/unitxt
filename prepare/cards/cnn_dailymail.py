from src.unitxt.blocks import (
    FormTask,
    InputOutputTemplate,
    LoadHF,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="cnn_dailymail", name="3.0.0"),
    task=FormTask(
        inputs=["article"],
        outputs=["highlights"],
        metrics=["metrics.rouge"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(input_format="{article}", output_format="{highlights}"),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.cnn_dailymail", overwrite=True)
