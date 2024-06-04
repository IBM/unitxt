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
    loader=LoadHF(path="glue", name="cola"),
    preprocess_steps=[
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "unacceptable", "1": "acceptable"}}),
        RenameFields(field_to_field={"sentence": "text"}),
        AddFields(
            fields={
                "classes": ["unacceptable", "acceptable"],
                "text_type": "text",
                "type_of_class": "grammatical acceptability",
            }
        ),
    ],
    task="tasks.classification.multi_class[metrics=[metrics.matthews_correlation]]",
    templates="templates.classification.multi_class.all",
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
        "Dataset Card for GLUE\n"
        "Dataset Summary\n"
        "GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems.\n"
        "Supported Tasks and Leaderboards\n"
        "The leaderboard for the GLUE benchmark can be found at this address. It comprises the following tasks:\n"
        "ax\n"
        "A manually-curated evaluation dataset for fine-grained… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
)

test_card(card, strict=False)
add_to_catalog(card, "cards.cola", overwrite=True)
