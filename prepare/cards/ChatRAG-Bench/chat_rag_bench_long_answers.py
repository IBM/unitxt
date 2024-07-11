from copy import deepcopy

from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
    TemplatesDict,
)
from unitxt.operators import Copy, Set
from unitxt.test_utils.card import test_card

subsets = ["doqa_travel", "doqa_cooking", "doqa_movies", "doc2dial", "hybridial"]
for subset in subsets:
    card = TaskCard(
        loader=LoadHF(path="nvidia/ChatRAG-Bench", name=subset, split="test"),
        preprocess_steps=[
            SplitRandomMix(
                {"train": "test[0.6]", "validation": "test[0.2]", "test": "test[0.2]"}
            ),
            Copy(
                field_to_field={
                    "ctxs/*/text": "contexts",
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
        task="tasks.rag.response_generation_multi_turn",
        templates=TemplatesDict(
            {"default": "templates.rag.response_generation.multi_turn.simple"}
        ),
    )

    # testing the card is too slow with the bert-score metric, so dropping it
    card_for_test = deepcopy(card)
    card_for_test.task.metrics = [
        "metrics.rouge",
    ]

    test_card(
        card_for_test,
        strict=True,
        demos_taken_from="test",
    )
    add_to_catalog(card, f"cards.chat_rag_bench.{subset}", overwrite=True)