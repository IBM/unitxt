from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Filter
from unitxt.operators import ListFieldValues, MapValues, SetEmptyDictIfDoesNotExist
from unitxt.processors import LiteralEval, Lower
from unitxt.splitters import RenameSplits
from unitxt.string_operators import MapReplace
from unitxt.test_utils.card import test_card

config_names = [
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

for name in config_names:
    card = TaskCard(
        loader=LoadHF(
            path="MMMU/MMMU", name=name, data_classification_policy=["public"]
        ),
        preprocess_steps=[
            RenameSplits(mapper={"dev": "train", "validation": "test"}),
            SetEmptyDictIfDoesNotExist(field="media"),
            ListFieldValues(
                fields=[f"image_{i}" for i in range(1, 8)], to_field="media/images"
            ),
            Filter(field="media/images", values=[None]),
            MapReplace(
                field_to_field={"question": "question", "options": "choices"},
                mapping={
                    f"<image {i}>": f'<img src="media/images/{i-1}">'
                    for i in range(1, 8)
                },
            ),
            LiteralEval(field="choices"),
            Lower(field="subfield", to_field="topic"),
            MapValues(
                field="answer",
                mapping={"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "?": None},
            ),
        ],
        task="tasks.qa.multiple_choice.with_topic",
        templates="templates.qa.multiple_choice.with_topic.all",
        __tags__={
            "language": ["en"],
            "license": "apache-2.0",
            "size_categories": ["10K<n<100K"],
            "task_categories": [
                "question-answering",
                "visual-question-answering",
                "multiple-choice",
            ],
        },
        __description__=(
            "MMMU: a new benchmark designed to evaluate multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes 11.5K meticulously collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions span 30 subjects and 183 subfields, comprising 30 highly heterogeneous image types, such as charts, diagrams, maps, tables, music sheets, and chemical structures. We believe MMMU will stimulate the community to build next-generation multimodal foundation models towards expert artificial general intelligence (AGI)."
        ),
    )
    if name == "Accounting":
        test_card(card, strict=False)
    add_to_catalog(card, f"cards.mmmu.{name.lower()}", overwrite=True)
