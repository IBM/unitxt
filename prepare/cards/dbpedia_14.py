import sys

from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from unitxt.operators import Shuffle
from unitxt.test_utils.card import test_card

dataset_name = "dbpedia_14"

classes = [
    "Company",
    "Educational Institution",
    "Artist",
    "Athlete",
    "Office Holder",
    "Mean Of Transportation",
    "Building",
    "Natural Place",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "Written Work",
]

mappers = {str(i): cls for i, cls in enumerate(classes)}

card = TaskCard(
    loader=LoadHF(path=f"{dataset_name}"),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        SplitRandomMix(
            {"train": "train[87.5%]", "validation": "train[12.5%]", "test": "test"}
        ),
        MapInstanceValues(mappers={"label": mappers}),
        RenameFields(field_to_field={"content": "text"}),
        AddFields(
            fields={
                "classes": classes,
                "text_type": "paragraph",
                "type_of_class": "topic",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
    __tags__={
        "annotations_creators": "machine-generated",
        "arxiv": "1509.01626",
        "croissant": True,
        "language": "en",
        "language_creators": "crowdsourced",
        "license": "cc-by-sa-3.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "100K<n<1M",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": "topic-classification",
    },
    __description__=(
        "Dataset Card for DBpedia14\n"
        "Dataset Summary\n"
        "The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes\n"
        "from DBpedia 2014. They are listed in classes.txt. From each of these 14 ontology classes, we\n"
        "randomly choose 40,000 training samples and 5,000 testing samples. Therefore, the total size\n"
        "of the training dataset is 560,000 and testing dataset 70,000.\n"
        "There are 3 columns in the dataset (same for train and test splits)… See the full description on the dataset page: https://huggingface.co/datasets/fancyzhx/dbpedia_14."
    ),
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
