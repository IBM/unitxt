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
    loader=LoadHF(path="wmt16", name="de-en"),
    preprocess_steps=[
        SplitRandomMix({"train": "train", "validation": "validation", "test": "test"}),
        CopyFields(field_to_field=[["translation/en", "en"], ["translation/de", "de"]], use_query=True),
    ],
    task=FormTask(
        inputs=["en"],
        outputs=["de"],
        metrics=["metrics.bleu"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="{en}\n",
                output_format="{de}",
            ),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.wmt_en_de", overwrite=True)
