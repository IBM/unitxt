from src.unitxt.blocks import (
    CopyFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="cnn_dailymail", name="3.0.0"),
    preprocess_steps=[
        # SplitRandomMix({"train": "train[95%]", "validation": "train[5%]", "test": "validation"}),
        # CopyFields(field_to_field=[["answers/text", "answer"]], use_query=True),
    ],
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
