import datasets as ds
from src.unitxt import dataset
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
from src.unitxt.splitters import RenameSplits
from src.unitxt.test_utils.card import test_card

for lang in [
    "fr",
    "vi",
    "zh",
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "en",
    "es",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
]:
    card = TaskCard(
        loader=LoadHF(path="xnli", name=lang),
        preprocess_steps=[
            RenameSplits({"validation_matched": "validation"}),
            "splitters.small_no_test",
            MapInstanceValues(mappers={"label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}),
            AddFields(
                fields={
                    "choices": ["entailment", "neutral", "contradiction"],
                }
            ),
        ],
        task="tasks.nli",
        templates="templates.classification.nli.all",
    )

    test_card(card)
    add_to_catalog(card, f"cards.xnli.{lang}", overwrite=True)
