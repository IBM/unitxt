from datasets import get_dataset_config_names
from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    RenameFields,
    TaskCard,
)
from unitxt.test_utils.card import test_card

dataset_name = "head_qa"

categories = [
    "biology",
    "chemistry",
    "medicine",
    "nursery",
    "pharmacology",
    "psychology",
]
for subset in get_dataset_config_names(dataset_name):
    card = TaskCard(
        loader=LoadHF(path=f"{dataset_name}", name=subset),
        preprocess_steps=[
            RenameFields(field_to_field={"qtext": "text", "category": "label"}),
            AddFields(
                fields={
                    "classes": categories,
                    "text_type": "question",
                    "type_of_class": "topic",
                }
            ),
        ],
        task="tasks.classification.multi_class",
        templates="templates.classification.multi_class.all",
        __description__=(
            "HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to access a specialized position in the Spanish healthcare system, and are challenging even for highly specialized humans. They are designed by the Ministerio de Sanidad, Consumo y Bienestar Social. The dataset contains questions about the following topics: medicine, nursing, psychology, chemistry, pharmacology and biology."
        ),
        __tags__={
            "annotations_creators": "no-annotation",
            "language": ["en", "es"],
            "language_creators": "expert-generated",
            "license": "mit",
            "multilinguality": "monolingual",
            "region": "us",
            "size_categories": "1K<n<10K",
            "source_datasets": "original",
            "task_categories": "question-answering",
            "task_ids": "multiple-choice-qa",
        },
    )
    test_card(card, debug=False)
    add_to_catalog(card, f"cards.{dataset_name}.{subset}", overwrite=True)
