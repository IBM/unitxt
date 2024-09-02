from copy import deepcopy

from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
    TemplatesDict,
)
from unitxt.dialog_operators import SerializeOpenAiFormatDialog
from unitxt.operators import Copy, Set, Shuffle
from unitxt.test_utils.card import test_card

subsets = ["doqa_travel", "doqa_cooking", "doqa_movies", "doc2dial", "hybridial"]
for subset in subsets:
    card = TaskCard(
        loader=LoadHF(path="nvidia/ChatRAG-Bench", name=subset, split="test"),
        preprocess_steps=[
            SplitRandomMix(
                {"train": "test[0.6]", "validation": "test[0.2]", "test": "test[0.2]"}
            ),
            Shuffle(),
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
            SerializeOpenAiFormatDialog(
                field="dialog",
                to_field="question",
                format="formats.user_assistant",
                slice_first_and_last_turns_format=True,
                last_response_to_field="dummy",
            ),
        ],
        task="tasks.rag.response_generation",
        templates=TemplatesDict(
            {"default": "templates.rag.response_generation.please_respond_chat"}
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
    add_to_catalog(
        card,
        f"cards.chat_rag_bench.train.user_assistant_format.{subset}",
        overwrite=True,
    )
