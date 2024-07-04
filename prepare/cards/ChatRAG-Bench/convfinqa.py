from copy import deepcopy

from unitxt import add_to_catalog
from unitxt.blocks import (
    SplitRandomMix,
    TaskCard,
    TemplatesDict,
)
from unitxt.loaders import LoadHF
from unitxt.operators import Copy, Set
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="nvidia/ChatRAG-Bench",
        data_files={"validation": "data/convfinqa/dev.json"},
    ),
    preprocess_steps=[
        SplitRandomMix(
            {
                "train": "validation[0.6]",
                "validation": "test[0.2]",
                "test": "validation[0.2]",
            }
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
    "metrics.rag.response_generation.correctness.token_overlap",
    "metrics.rag.response_generation.faithfullness.token_overlap",
]

test_card(
    card_for_test,
    strict=True,
    demos_taken_from="test",
)
add_to_catalog(card, "cards.chat_rag_bench.convfinqa", overwrite=True)
