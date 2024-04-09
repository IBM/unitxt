from unitxt.blocks import AddFields, LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import CopyFields, IndexOf, RenameFields
from unitxt.test_utils.card import test_card

subtasks = ["ARC-Challenge", "ARC-Easy"]

for subtask in subtasks:
    card = TaskCard(
        loader=LoadHF(path="ai2_arc", name=subtask),
        preprocess_steps=[
            AddFields({"topic": "science"}),
            RenameFields(field_to_field={"answerKey": "label", "choices": "_choices"}),
            CopyFields(
                field_to_field={"_choices/text": "choices", "_choices/label": "labels"}
            ),
            IndexOf(search_in="labels", index_of="label", to_field="answer"),
        ],
        task="tasks.qa.multiple_choice.with_topic",
        templates="templates.qa.multiple_choice.with_topic.all",
    )
    test_card(card, strict=False)
    add_to_catalog(
        card, f'cards.ai2_arc.{subtask.replace("-", "_").lower()}', overwrite=True
    )
