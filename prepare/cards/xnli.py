from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

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
            RenameFields(field_to_field={"premise": "text_a", "hypothesis": "text_b"}),
            MapInstanceValues(
                mappers={
                    "label": {"0": "entailment", "1": "neutral", "2": "contradiction"}
                }
            ),
            AddFields(
                fields={
                    "type_of_relation": "entailment",
                    "text_a_type": "premise",
                    "text_b_type": "hypothesis",
                    "classes": ["entailment", "neutral", "contradiction"],
                }
            ),
        ],
        task="tasks.classification.multi_class.relation",
        templates="templates.classification.multi_class.relation.all",
        __tags__={
            "croissant": True,
            "language": [
                "ar",
                "bg",
                "de",
                "el",
                "en",
                "es",
                "fr",
                "hi",
                "ru",
                "sw",
                "th",
                "tr",
                "ur",
                "vi",
                "zh",
            ],
            "region": "us",
        },
    )
    if lang == langs[0]:
        test_card(card)
    add_to_catalog(card, f"cards.xnli.{lang}", overwrite=True)
