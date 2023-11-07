from typing import List, Union

from src.unitxt.blocks import AddFields, FormTask, InputOutputTemplate, LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operator import StreamingOperator
from src.unitxt.operators import JoinStr, RenameFields, TakeByField, ZipFieldValues
from src.unitxt.splitters import RenameSplits
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


def multiple_choice_outputs():
    return ["label"]


def multiple_choice_inputs_outputs(context=False):
    return {"inputs": multiple_choice_inputs(context=context), "outputs": multiple_choice_outputs()}


def multiple_choice_inputs(context=False):
    inputs = ["choices", "sentence1", "numbers", "topic"]
    if context:
        inputs.append("context")
    return inputs


def multiple_choice_preprocess(
    question: str,
    numbering: str,
    choices: str,
    topic: str,
    label_index: str,
    context: str = None,
    expected_answer: str = "number",
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
    input_fields = [numbering, choices, label_index]
    renames = {field: "_" + field for field in input_fields}
    renames[topic] = "topic"
    if context:
        renames[context] = "context"
    renames[question] = "sentence1"
    return [
        RenameFields(field_to_field=renames),
        TakeByField(field=renames[numbering], index=renames[label_index], to_field="number"),
        TakeByField(field=renames[choices], index=renames[label_index], to_field="answer"),
        ZipFieldValues(fields=[renames[numbering], renames[choices]], to_field="choices"),
        JoinStr(separator=". ", field="choices/*", to_field="choices_list", use_query=True, process_every_value=True),
        TakeByField(field="choices_list", index=renames[label_index], to_field="number_and_answer"),
        JoinStr(separator=",", field="choices/*/0", to_field="numbers", use_query=True),
        JoinStr(separator=" ", field="choices_list", to_field="choices"),  # field_to_field
        RenameFields(field_to_field={expected_answer: "label"}),
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
                AddFields({"numbering": numbering, "topic": subtask.replace("_", " ")}),
                *multiple_choice_preprocess(
                    question="question",
                    numbering="numbering",
                    choices="choices",
                    topic="topic",
                    label_index="answer",
                    expected_answer=expected_answer,
                ),
            ],
            task=FormTask(
                **multiple_choice_inputs_outputs(),
                metrics=["metrics.accuracy"],
            ),
            templates="templates.qa.multiple_choice.original.all",
        )
        test_card(card)
        add_to_catalog(card, f"cards.mmlu.{subtask}", overwrite=True)


if __name__ == "__main__":
    main()
