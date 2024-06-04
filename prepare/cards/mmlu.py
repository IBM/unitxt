from unitxt.blocks import AddFields, LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

subtasks = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def main():
    for i, subtask in enumerate(subtasks):
        card = TaskCard(
            loader=LoadHF(path="cais/mmlu", name=subtask),
            preprocess_steps=[
                RenameSplits({"dev": "train"}),
                AddFields({"topic": subtask.replace("_", " ")}),
            ],
            task="tasks.qa.multiple_choice.with_topic",
            templates="templates.qa.multiple_choice.with_topic.all",
            __tags__={
                "annotations_creators": "no-annotation",
                "arxiv": ["2009.03300", "2005.00700", "2005.14165", "2008.02275"],
                "flags": ["croissant"],
                "language": "en",
                "language_creators": "expert-generated",
                "license": "mit",
                "multilinguality": "monolingual",
                "region": "us",
                "size_categories": "10K<n<100K",
                "source_datasets": "original",
                "task_categories": "question-answering",
                "task_ids": "multiple-choice-qa",
            },
            __description__=(
                "Dataset Card for MMLU Dataset Summary Measuring Massive Multitask Language Understanding by Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021). This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge. The test spans subjects in the humanities, social sciences, hard sciences, and other areas that are important for some people to learn. This covers 57… See the full description on the dataset page: https://huggingface.co/datasets/cais/mmlu."
            ),
        )
        if i == 0:
            test_card(card, strict=False)
        add_to_catalog(card, f"cards.mmlu.{subtask}", overwrite=True)


main()
