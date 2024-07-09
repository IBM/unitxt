from datasets import load_dataset_builder
from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    MapInstanceValues,
    Set,
    TaskCard,
)
from unitxt.test_utils.card import test_card

dataset_name = "ledgar"

ds_builder = load_dataset_builder("lex_glue", dataset_name)
classlabels = ds_builder.info.features["label"]

mappers = {}
for i in range(len(classlabels.names)):
    mappers[str(i)] = classlabels.names[i]

card = TaskCard(
    loader=LoadHF(path="lex_glue", name=f"{dataset_name}"),
    preprocess_steps=[
        MapInstanceValues({"label": mappers}),
        Set(
            fields={
                "classes": classlabels.names,
                "type_of_class": "contractual clauses",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
    __tags__={
        "annotations_creators": "found",
        "arxiv": ["2110.00976", "2109.00904", "1805.01217", "2104.08671"],
        "language": "en",
        "language_creators": "found",
        "license": "cc-by-4.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "extended",
        "task_categories": ["question-answering", "text-classification"],
        "task_ids": [
            "multi-class-classification",
            "multi-label-classification",
            "multiple-choice-qa",
            "topic-classification",
        ],
    },
    __description__=(
        "LEDGAR dataset aims contract provision (paragraph) classification. The contract provisions come from contracts obtained from the US Securities and Exchange Commission (SEC) filings, which are publicly available from EDGAR. Each label represents the single main topic (theme) of the corresponding contract provision… See the full description on the dataset page: https://huggingface.co/datasets/coastalcph/lex_glue."
    ),
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
