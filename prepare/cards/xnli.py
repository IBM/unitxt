import datasets as ds
from unitxt import dataset
from unitxt.blocks import (
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
from unitxt.catalog import add_to_catalog
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

for lang in ['fr']:#, 'vi', 'zh', 'ar', 'bg', 'de', 'el', 'en', 'en', 'es', 'hi', 'ru', 'sw', 'th', 'tr', 'ur']:
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
    add_to_catalog(card, f"cards.xnli_{lang}", overwrite=True)
