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

card = TaskCard(
    loader=LoadHF(path="glue", name="mnli"),
    preprocess_steps=[
        RenameSplits({"validation_matched": "validation"}),
        "splitters.small_no_test",
        RenameFields(field_to_field={"premise": "text_a", "hypothesis": "text_b"}),
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}
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
        "annotations_creators": "other",
        "arxiv": "1804.07461",
        "flags": [
            "coreference-nli",
            "croissant",
            "paraphrase-identification",
            "qa-nli",
        ],
        "language": "en",
        "language_creators": "other",
        "license": "other",
        "multilinguality": "monolingual",
        "region": "us",
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
        "Dataset Card for GLUE Dataset Summary GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems. Supported Tasks and Leaderboards The leaderboard for the GLUE benchmark can be found at this address. It comprises the following tasks: ax A manually-curated evaluation dataset for fine-grained… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
)

test_card(card)
add_to_catalog(card, "cards.mnli", overwrite=True)
