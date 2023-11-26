from src.unitxt.blocks import AddFields, LoadHF, MapInstanceValues, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.splitters import RenameSplits
from src.unitxt.test_utils.card import test_card

langs = [
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
]

for lang in langs:
    card = TaskCard(
        loader=LoadHF(path="xnli", name=lang),
        preprocess_steps=[
            RenameSplits({"validation_matched": "validation"}),
            "splitters.small_no_test",
            MapInstanceValues(
                mappers={
                    "label": {"0": "entailment", "1": "neutral", "2": "contradiction"}
                }
            ),
            AddFields(
                fields={
                    "choices": ["entailment", "neutral", "contradiction"],
                }
            ),
        ],
        task="tasks.nli",
        templates="templates.classification.nli.all",
    )
    if lang == lang[0]:
        test_card(card)
    add_to_catalog(card, f"cards.xnli.{lang}", overwrite=True)
