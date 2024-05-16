from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    AddFields,
    CopyFields,
    RenameFields,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="dim/mt_bench_en", split="train"),
    preprocess_steps=[
        RenameSplits({"train": "test"}),
        CopyFields(field_to_field={"turns/0": "turns"}),
        RenameFields(
            field_to_field={
                "turns": "input",
                "category": "group",
            }
        ),
        AddFields(
            fields={
                "output": "None",
                "type_of_input": "question",
                "type_of_output": "answer",
            }
        ),
    ],
    task="tasks.generation",
    templates=["templates.empty"],
)

test_card(card, demos_taken_from="test", strict=False)
add_to_catalog(
    card,
    "cards.mt_bench.generation.english_single_turn",
    overwrite=True,
)
