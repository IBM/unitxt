from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="wnli"),
    preprocess_steps=[
        "splitters.small_no_test",
        RenameFields(
            field_to_field={
                "sentence1": "text_a",
                "sentence2": "text_b",
            }
        ),
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        AddFields(
            fields={
                "classes": ["entailment", "not entailment"],
                "type_of_relation": "entailment",
                "text_a_type": "premise",
                "text_b_type": "hypothesis",
            }
        ),
    ],
    task="tasks.classification.multi_class.relation",
    templates="templates.classification.multi_class.relation.all",
    __tags__={
        "dataset_info_tags": [
            "task_categories:text-classification",
            "task_ids:acceptability-classification",
            "task_ids:natural-language-inference",
            "task_ids:semantic-similarity-scoring",
            "task_ids:sentiment-classification",
            "task_ids:text-scoring",
            "annotations_creators:other",
            "language_creators:other",
            "multilinguality:monolingual",
            "size_categories:10K<n<100K",
            "source_datasets:original",
            "language:en",
            "license:other",
            "qa-nli",
            "coreference-nli",
            "paraphrase-identification",
            "croissant",
            "arxiv:1804.07461",
            "region:us",
        ]
    },
)

test_card(card)
add_to_catalog(card, "cards.wnli", overwrite=True)


card = TaskCard(
    loader=LoadHF(path="glue", name="wnli"),
    preprocess_steps=[
        "splitters.small_no_test",
        RenameFields(
            field_to_field={
                "sentence1": "text_a",
                "sentence2": "text_b",
            }
        ),
        MapInstanceValues(mappers={"label": {"0": "yes", "1": "no"}}),
        AddFields(
            fields={
                "classes": ["yes", "no"],
                "type_of_relation": "truthfulness",
                "text_a_type": "premise",
                "text_b_type": "hypothesis",
            }
        ),
    ],
    task="tasks.classification.multi_class.relation",
    templates="templates.classification.multi_class.relation.truthfulness.all",
    __tags__={
        "dataset_info_tags": [
            "task_categories:text-classification",
            "task_ids:acceptability-classification",
            "task_ids:natural-language-inference",
            "task_ids:semantic-similarity-scoring",
            "task_ids:sentiment-classification",
            "task_ids:text-scoring",
            "annotations_creators:other",
            "language_creators:other",
            "multilinguality:monolingual",
            "size_categories:10K<n<100K",
            "source_datasets:original",
            "language:en",
            "license:other",
            "qa-nli",
            "coreference-nli",
            "paraphrase-identification",
            "croissant",
            "arxiv:1804.07461",
            "region:us",
        ]
    },
)

test_card(card)
add_to_catalog(card, "cards.wnli.truthfulness", overwrite=True)
