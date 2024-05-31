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
        "annotations_creators": "other",
        "arxiv": "1804.07461",
        "language": "en",
        "language_creators": "other",
        "license": "other",
        "multilinguality": "monolingual",
        "region": "us",
        "singletons": [
            "coreference-nli",
            "croissant",
            "paraphrase-identification",
            "qa-nli",
        ],
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": [
            "acceptability-classification",
            "natural-language-inference",
            "semantic-similarity-scoring",
            "sentiment-classification",
            "text-scoring",
        ],
    },
    __description__=(
        "Dataset Card for GLUE\n"
        "Dataset Summary\n"
        "GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems.\n"
        "Supported Tasks and Leaderboards\n"
        "The leaderboard for the GLUE benchmark can be found at this address. It comprises the following tasks:\n"
        "ax\n"
        "A manually-curated evaluation dataset for fine-grained… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
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
        "annotations_creators": "other",
        "arxiv": "1804.07461",
        "language": "en",
        "language_creators": "other",
        "license": "other",
        "multilinguality": "monolingual",
        "region": "us",
        "singletons": [
            "coreference-nli",
            "croissant",
            "paraphrase-identification",
            "qa-nli",
        ],
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": [
            "acceptability-classification",
            "natural-language-inference",
            "semantic-similarity-scoring",
            "sentiment-classification",
            "text-scoring",
        ],
    },
    __description__=(
        "Dataset Card for GLUE\n"
        "Dataset Summary\n"
        "GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems.\n"
        "Supported Tasks and Leaderboards\n"
        "The leaderboard for the GLUE benchmark can be found at this address. It comprises the following tasks:\n"
        "ax\n"
        "A manually-curated evaluation dataset for fine-grained… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
)

test_card(card)
add_to_catalog(card, "cards.wnli.truthfulness", overwrite=True)
