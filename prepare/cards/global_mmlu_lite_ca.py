from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    Deduplicate,
    ListFieldValues,
    MapInstanceValues,
    Rename,
)
from unitxt.settings_utils import get_settings
from unitxt.splitters import SplitRandomMix
from unitxt.test_utils.card import test_card

languages = [
    "ar",
    "bn",
    "de",
    "fr",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "pt",
    "es",
    "sw",
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
subject_mapping = {subject: subject.replace("_", " ") for subject in subtasks}

is_first = True
settings = get_settings()
with settings.context(allow_unverified_code=True):
    for language in languages:
        card = TaskCard(
            loader=LoadHF(
                path="CohereForAI/Global-MMLU-Lite",
                name=language,
                filtering_lambda=f"lambda x: x['cultural_sensitivity_label'] == 'CA'",
            ),
            preprocess_steps=[
                SplitRandomMix({"test": "test[100%]", "train": "test[10%]"}),
                Deduplicate(by=["question", "subject", "answer"]),
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
                Rename(field_to_field={"subject": "topic"}),
                MapInstanceValues(mappers={"topic": subject_mapping}),
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
                "Global-MMLU-Lite is a streamlined multilingual evaluation set covering 15 languages. The dataset "
                "includes 200 Culturally Sensitive (CS) and 200 Culturally Agnostic (CA) questions per language. "
                "The samples in Global-MMLU-Lite correspond to languages that were fully human-translated or "
                "post-edited in the original dataset. This initiative was led by Cohere For AI in collaboration "
                "with external contributors from industry and academia. The test spans subjects in humanities, "
                "social sciences, hard sciences, and other areas. For more information, see: "
                "https://huggingface.co/datasets/CohereForAI/Global-MMLU-Lite"
            ),
        )

        if is_first:
            test_card(card, strict=False)
            is_first = False
        add_to_catalog(card, f"cards.global_mmlu_lite_ca.{language}", overwrite=True)
