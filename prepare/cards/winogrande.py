from datasets import load_dataset_builder
from prepare.cards.mmlu import (
    MMLU_TEMPLATES,
    multiple_choice_inputs_outputs,
    multiple_choice_preprocess,
)
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
from src.unitxt.operators import (
    AddConstant,
    CastFields,
    CopyFields,
    IndexOf,
    JoinStr,
    ListFieldValues,
    RenameFields,
    TakeByField,
    ZipFieldValues,
)
from src.unitxt.test_utils.card import test_card

subtasks = ["debiased", "l", "m", "s", "xl", "xs"]

for subtask in subtasks:
    # numbering=tuple(str(x) for x in range(200))
    numbering = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    expected_answer = "number"  # 'number_and_answer' #'number'

    card = TaskCard(
        loader=LoadHF(path="winogrande", name=f"winogrande_{subtask}"),
        preprocess_steps=[
            "splitters.small_no_test",
            AddFields({"topic": "common sense", "numbering": numbering}),
            ListFieldValues(fields=["option1", "option2"], to_field="choices"),
            CastFields(fields={"answer": "int"}),
            AddConstant(field="answer", add=-1),
            *multiple_choice_preprocess(
                question="sentence", numbering="numbering", choices="choices", topic="topic", label_index="answer"
            ),
        ],
        task=FormTask(
            **multiple_choice_inputs_outputs(),
            metrics=["metrics.accuracy"],
        ),
        templates=MMLU_TEMPLATES,
    )
    if subtask == subtask[0]:
        test_card(card, demos_taken_from="test")
    add_to_catalog(card, f"cards.winogrande.{subtask.replace('-', '_')}", overwrite=True)
