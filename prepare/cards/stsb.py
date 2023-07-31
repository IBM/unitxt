from src.unitxt.blocks import (
    FormTask,
    LoadHF,
    OutputQuantizingTemplate,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="stsb"),
    preprocess_steps=[
        SplitRandomMix({"train": "train[95%]", "validation": "train[5%]", "test": "validation"}),
    ],
    task=FormTask(
        inputs=["sentence1", "sentence2"],
        outputs=["label"],
        metrics=["metrics.spearman"],
    ),
    templates=TemplatesList(
        [
            OutputQuantizingTemplate(
                input_format="""
                   Given this sentence: '{sentence1}', on a scale of 1 to 5, how similar in meaning is it to this sentence: '{sentence2}'?
                """.strip(),
                output_format="{label}",
                quantum=0.2,
            ),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.stsb", overwrite=True)
