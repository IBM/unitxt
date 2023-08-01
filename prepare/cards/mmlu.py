from typing import List, Union

from datasets import load_dataset_builder
from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    NormalizeListFields,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operator import StreamingOperator
from src.unitxt.operators import JoinStr, RenameFields, TakeByField, ZipFieldValues
from src.unitxt.splitters import RenameSplits
from src.unitxt.templates import TemplatesDict
from src.unitxt.test_utils.card import test_card

# import huggingface_hub
# from huggingface_hub.hf_api import DatasetInfo as HFDatasetInfo, HfApi
# from huggingface_hub import DatasetFilter
# api = HfApi()
# analyzer = AnalyzerEngine()
# datasets = list(api.list_datasets(filter=DatasetFilter(dataset_name='cais/mmlu')))
# builder = load_dataset_builder(path='cais/mmlu')
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
MMLU_TEMPLATES = TemplatesDict(
    {
        "original": InputOutputTemplate(
            input_format="""
                            The following are multiple choice questions (with answers) about {topic}.\n
                            {sentence1}.\nAnswers: {choices}.\nAnswer:
                    """.strip(),
            output_format="{label}",
        ),
        "helm": InputOutputTemplate(
            input_format="""
                            The following are multiple choice questions (with answers) about {topic}.\n\n
                            Question: {sentence1}.\nAnswers: {choices}.\nAnswer:
                    """.strip(),
            output_format="{label}",
        ),
        "lm_eval_harness": InputOutputTemplate(
            input_format="""
                            Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:
                    """.strip(),
            output_format="{label}",
        ),
        "fm-eval": InputOutputTemplate(
            input_format="""
                            The following are multiple choice questions (with answers) about {topic}.\n\n
                            Question: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:
                    """.strip(),
            output_format="{label}",
        ),
    }
)


def multiple_choice_preprocess(
    numbering: str, choices: str, topic: str, label_index: str, expected_answer: str = "number"
) -> List[Union[StreamingOperator, str]]:
    """
    Processing to make a unified format of multiple choice questions
    :param numbering: the field containing the numerals to use (e.g. ABCD [1,2,3,4])
    :param choices: the field with the choices (e.g. ['apple','bannana']
    :param topic: the field containing the topic of the question
    :param label_index: in what index is the right index (consider using IndexOf function if you have the answer instead)
    :param expected_answer: what format should the 'label' field be answer\number\number_and_answer
    :return:
    """

    assert expected_answer in ["number", "number_and_answer", "answer"]
    return [
        TakeByField(field=numbering, index=label_index, to_field="number"),
        TakeByField(field=choices, index=label_index, to_field="answer"),
        ZipFieldValues(fields=[numbering, choices], to_field="choices"),
        JoinStr(separator=". ", field="choices/*", to_field="choices_list", use_query=True, process_every_value=True),
        TakeByField(field="choices_list", index=label_index, to_field="number_and_answer"),
        JoinStr(separator=",", field="choices/*/0", to_field="numbers", use_query=True),
        JoinStr(separator=" ", field="choices_list", to_field="choices"),  # field_to_field
        RenameFields({expected_answer: "label"}),
    ]


def main():
    for subtask in subtasks:
        # numbering=tuple(str(x) for x in range(200))
        numbering = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        expected_answer = "number"  # "number_and_answer" #"number"

        card = TaskCard(
            loader=LoadHF(path="cais/mmlu", name=subtask),
            preprocess_steps=[
                RenameSplits({"auxiliary_train": "train"}),
                RenameFields({"answer": "label", "question": "sentence1"}),
                AddFields({"numbering": numbering, "topic": subtask.replace("_", " ")}),
                *multiple_choice_preprocess(
                    numbering="numbering",
                    choices="choices",
                    topic="topic",
                    label_index="label",
                    expected_answer=expected_answer,
                ),
            ],
            task=FormTask(
                inputs=["choices", "sentence1", "topic", "numbers"],
                outputs=[
                    "label",
                ],
                metrics=["metrics.accuracy"],
            ),
            templates=MMLU_TEMPLATES,
        )
        test_card(card)
        add_to_catalog(card, f"cards.mmlu.{subtask}", overwrite=True)


if __name__ == "__main__":
    main()
