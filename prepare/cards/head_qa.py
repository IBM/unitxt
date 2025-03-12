import unitxt
from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    Rename,
    Set,
    TaskCard,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

with unitxt.settings.context(allow_unverified_code=True):
    for subset in ["es", "en", "gl", "it", "ru"]:
        card = TaskCard(
            loader=LoadHF(path="alesi12/head_qa_v2", name=subset),
            preprocess_steps=[
                RenameSplits({"train": "test"}),
                Rename(field_to_field={"qtext": "text", "category": "label"}),
                Set(
                    fields={
                        "classes": [
                            "biology",
                            "chemistry",
                            "medicine",
                            "nursery",
                            "pharmacology",
                            "psychology",
                        ],
                        "text_type": "question",
                    }
                ),
            ],
            task="tasks.classification.multi_class.topic_classification",
            templates="templates.classification.multi_class.all",
            __description__=(
                "HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to access a specialized position in the Spanish healthcare system, and are challenging even for highly specialized humans. They are designed by the Ministerio de Sanidad, Consumo y Bienestar Social. The dataset contains questions about the following topics: medicine, nursing, psychology, chemistry, pharmacology and biology… See the full description on the dataset page: https://huggingface.co/datasets/head_qa"
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
        if subset == "es":
            test_card(card, debug=False)
        add_to_catalog(card, f"cards.head_qa.{subset}", overwrite=True)
