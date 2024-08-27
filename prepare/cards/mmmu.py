from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.image_operators import ImageToText
from unitxt.operators import ListFieldValues, Rename, ZipFieldValues
from unitxt.splitters import RenameSplits
from unitxt.string_operators import FindAll
from unitxt.test_utils.card import test_card

topics = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]

for topic in topics:
    card = TaskCard(
        loader=LoadHF(path="MMMU/MMMU", name=topic),
        preprocess_steps=[
            RenameSplits({"dev": "train", "validation": "validation", "test": "test"}),
            Rename(field_to_field={"options": "choices"}),
            Set(fields={"topic": topic.lower().replace("_", " ")}),
            FindAll(
                pattern=r"<(image_\d+)>", field="question", to_field="images_fields"
            ),
            ListFieldValues(
                fields=[
                    "image_1",
                    "image_2",
                    "image_3",
                    "image_4",
                    "image_5",
                    "image_6",
                    "image_7",
                ],
                to_field="images",
            ),
            ZipFieldValues(fields=["images", "images"]),
            ImageToText(field="image", to_field="context"),
        ],
        task="tasks.qa.multiple_choice.with_topic",
        templates="templates.qa.multiple_choice.with_topic.all",
        __tags__={
            "license": "apache-2.0",
            "multilinguality": "monolingual",
            "modalities": ["image", "text"],
            "size_categories": "10K<n<100K",
            "task_categories": "question-answering",
            "task_ids": "extractive-qa",
        },
        __description__=(
            "The doc-vqa Dataset integrates images from the Infographic_vqa dataset sourced from HuggingFaceM4 The Cauldron dataset, as well as images from the dataset AFTDB (Arxiv Figure Table Database) curated by cmarkea. This dataset consists of pairs of images and corresponding text, with each image linked to an average of five questions and answers available in both English and French. These questions and answers were generated using Gemini 1.5 Pro, thereby rendering the dataset well-suited for multimodal tasks involving image-text pairing and multilingual question answering."
        ),
    )

    test_card(card)
    add_to_catalog(card, f"cards.mmmu.{topic}", overwrite=True)
