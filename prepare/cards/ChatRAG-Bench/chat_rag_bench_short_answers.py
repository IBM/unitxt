from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
    TemplatesDict,
)
from unitxt.operators import Copy, RenameFields, Set
from unitxt.test_utils.card import test_card

subsets = ["sqa"]
for subset in subsets:
    card = TaskCard(
        loader=LoadHF(path="nvidia/ChatRAG-Bench", name=subset, split="test"),
        preprocess_steps=[
            SplitRandomMix(
                {"train": "test[0.4]", "validation": "test[0.25]", "test": "test[0.35]"}
            ),
            Copy(
                field_to_field={
                    "ctxs/*/text": "contexts",
                }
            ),
            RenameFields(
                field_to_field={
                    "messages": "dialog",
                    "answers": "reference_answers",
                }
            ),
            Set(
                fields={
                    "contexts_ids": [],
                }
            ),
        ],
        task="tasks.rag.response_generation_multi_turn_f1",
        templates=TemplatesDict(
            {
                "default": "templates.rag.response_generation.multi_turn.simple_short_answers"
            }
        ),
    )

    test_card(
        card,
        strict=True,
        demos_taken_from="test",
    )
    add_to_catalog(card, f"cards.chat_rag_bench.{subset}", overwrite=True)
