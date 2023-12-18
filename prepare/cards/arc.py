from prepare.cards.mmlu_old import (
    multiple_choice_preprocess,
)
from src.unitxt.blocks import (
    AddFields,
    FormTask,
    LoadHF,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import (
    CopyFields,
    IndexOf,
    RenameFields,
)
from src.unitxt.test_utils.card import test_card

subtasks = ["ARC-Challenge", "ARC-Easy"]

for subtask in subtasks:
    # numbering=tuple(str(x) for x in range(200))
    numbering = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    expected_answer = "number"  # 'number_and_answer' #'number'

    card = TaskCard(
        loader=LoadHF(path="ai2_arc", name=subtask),
        preprocess_steps=[
            AddFields({"topic": "science"}),
            RenameFields(
                field_to_field={"answerKey": "label", "choices": "choices_struct"}
            ),
            CopyFields(
                field_to_field={
                    "choices_struct/text": "choices",
                    "choices_struct/label": "numbering",
                },
                use_query=True,
            ),
            IndexOf(search_in="numbering", index_of="label", to_field="index"),
            *multiple_choice_preprocess(
                question="question",
                numbering="numbering",
                choices="choices",
                topic="topic",
                label_index="index",
            ),
        ],
        task=FormTask(
            inputs=["choices", "sentence1", "numbers", "topic"],
            outputs=[
                "label",
            ],
            metrics=["metrics.accuracy"],
        ),
        templates="templates.qa.multiple_choice.original.all",
    )
    test_card(card)
    add_to_catalog(card, f'cards.ai2_arc.{subtask.replace("-", "_")}', overwrite=True)
