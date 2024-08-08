from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    Copy,
    Rename,
    Set,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="shi3z/MTbenchJapanese", split="train"),
    preprocess_steps=[
        RenameSplits({"train": "test"}),
        Copy(field="turns/0", to_field="turns"),
        Rename(
            field_to_field={
                "turns": "input",
                "category": "group",
            }
        ),
        Set(
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
    "cards.mt_bench.generation.japanese_single_turn",
    overwrite=True,
)
