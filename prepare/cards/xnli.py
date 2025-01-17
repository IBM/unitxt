from unitxt.blocks import (
    LoadHF,
    MapInstanceValues,
    Rename,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
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
            "splitters.small_no_test",
            Rename(field_to_field={"premise": "text_a", "hypothesis": "text_b"}),
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
            "XNLI is a subset of MNLI with a few thousand examples translated into 14 different languages, including some low-resource ones. The task is to predict textual entailment, classifying whether sentence A implies, contradicts, or is neutral with respect to sentence B."
        ),
        __short_description__=(
            "Uses a subset of data from the MNLI (Multi-Genre Natural Language Inference) dataset, which has crowd-sourced sentence pairs that are annotated with textual entailment information, translated into 14 languages to evaluate how well a model can classify multilingual sentences. Metric measures the accuracy of answers."
        ),
    )
    if lang == langs[0]:
        test_card(card)
    add_to_catalog(card, f"cards.xnli.{lang}", overwrite=True)
