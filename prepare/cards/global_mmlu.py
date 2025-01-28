from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    Deduplicate,
    FilterByCondition,
    ListFieldValues,
    MapInstanceValues,
    Set,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

languages = [
    "am",
    "ar",
    "bn",
    "cs",
    "de",
    "el",
    "en",
    "es",
    "fa",
    "fil",
    "fr",
    "ha",
    "he",
    "hi",
    "id",
    "ig",
    "it",
    "ja",
    "ko",
    "ky",
    "lt",
    "mg",
    "ms",
    "ne",
    "nl",
    "ny",
    "pl",
    "pt",
    "ro",
    "ru",
    "si",
    "sn",
    "so",
    "sr",
    "sv",
    "sw",
    "te",
    "tr",
    "uk",
    "vi",
    "yo",
    "zh",
]
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


is_first = True
for language in languages:
    for subject in subtasks:
        card = TaskCard(
            loader=LoadHF(path="CohereForAI/Global-MMLU", name=language),
            preprocess_steps=[
                FilterByCondition(values={"subject": subject}, condition="eq"),
                Deduplicate(by=["question", "subject", "answer"]),
                RenameSplits({"dev": "train"}),
                MapInstanceValues(
                    mappers={
                        "answer": {
                            "A": 0,
                            "B": 1,
                            "C": 2,
                            "D": 3,
                        }
                    }
                ),
                ListFieldValues(
                    fields=["option_a", "option_b", "option_c", "option_d"],
                    to_field="choices",
                ),
                Set({"topic": subject.replace("_", " ")}),
            ],
            task="tasks.qa.multiple_choice.with_topic",
            templates="templates.qa.multiple_choice.with_topic.all",
            __tags__={
                "annotations_creators": "expert-generated",
                "language": language,
                "language_creators": "expert-generated",
                "license": "apache-2.0",
                "multilinguality": "multilingual",
                "size_categories": "10K<n<100K",
                "source_datasets": "original",
                "task_categories": "question-answering",
                "task_ids": "multiple-choice-qa",
                "region": "global",
            },
            __description__=(
                "Global-MMLU is a multilingual evaluation set spanning 42 languages, combining machine translations "
                "for MMLU questions along with professional translations and crowd-sourced post-edits. The dataset "
                "includes cultural sensitivity annotations, classifying questions as Culturally Sensitive (CS) or "
                "Culturally Agnostic (CA)️. This initiative was led by Cohere For AI in collaboration with external "
                "contributors from industry and academia. The test spans subjects in humanities, social sciences, hard "
                "sciences, and other areas. See the full description on the dataset page: "
                "https://huggingface.co/datasets/CohereForAI/Global-MMLU"
            ),
        )

        if is_first:
            test_card(card, strict=False)
            is_first = False
        add_to_catalog(card, f"cards.global_mmlu.{language}.{subject}", overwrite=True)
