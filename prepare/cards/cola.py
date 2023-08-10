from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    NormalizeListFields,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card
import datasets as ds

card = TaskCard(
    loader=LoadHF(path="glue", name="cola"),
    preprocess_steps=[
        SplitRandomMix({"train": "train[5%:95%]", "validation": "train[:5%]+train[95%:]", "test": "validation"}),
        MapInstanceValues(mappers={"label": {"0": "unacceptable", "1": "acceptable"}}),
        AddFields(
            fields={
                "choices": ["unacceptable", "acceptable"],
            }
        ),
    ],
    task=FormTask(
        inputs=["choices", "sentence"],
        outputs=["label"],
        metrics=["metrics.matthews_correlation"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="""
                    Given this sentence: {sentence}, classify if it is {choices}.
                """.strip(),
                output_format="{label}",
            ),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.cola", overwrite=True)

dataset = ds.load_dataset("card=cards.cola")
print()