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
                "dataset_info_tags": [
                    "task_categories:question-answering",
                    "task_ids:multiple-choice-qa",
                    "annotations_creators:no-annotation",
                    "language_creators:expert-generated",
                    "multilinguality:monolingual",
                    "size_categories:10K<n<100K",
                    "source_datasets:original",
                    "language:en",
                    "license:mit",
                    "croissant",
                    "arxiv:2009.03300",
                    "arxiv:2005.00700",
                    "arxiv:2005.14165",
                    "arxiv:2008.02275",
                    "region:us",
                ]
            },
        )
        if i == 0:
            test_card(card, strict=False)
        add_to_catalog(card, f"cards.mmlu.{subtask}", overwrite=True)


main()
