from unitxt.blocks import (
    LoadHF,
    MapInstanceValues,
    RenameFields,
    Set,
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
            Set(
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
        __description__=(
            "XNLI is a subset of a few thousand examples from MNLI which has been translated into 14 different languages (some low-ish resource). As with MNLI, the goal is to predict textual entailment (does sentence A imply/contradict/neither sentence B) and is a classification task (given two sentences, predict one of three labels)… See the full description on the dataset page: https://huggingface.co/datasets/xnli."
        ),
    )
    if lang == langs[0]:
        test_card(card)
    add_to_catalog(card, f"cards.xnli.{lang}", overwrite=True)
