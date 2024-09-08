from copy import deepcopy

from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
    TemplatesDict,
)
from unitxt.operators import (
    Copy,
    ListFieldValues,
    Shuffle,
)
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="umarbutler/open-australian-legal-qa",
    ),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[0.5]", "validation": "train[0.2]", "test": "train[0.3]"}
        ),
        Shuffle(),
        Copy(
            field_to_field={
                "source/text": "contexts",
                "answer": "reference_answers",
                "source/citation": "contexts_ids",
            }
        ),
        ListFieldValues(fields=["reference_answers"], to_field="reference_answers"),
        ListFieldValues(fields=["contexts"], to_field="contexts"),
        ListFieldValues(fields=["contexts_ids"], to_field="contexts_ids"),
    ],
    task="tasks.rag.response_generation",
    templates=TemplatesDict(
        {"default": "templates.rag.response_generation.please_respond_chat"}
    ),
)

# testing the card is too slow with the bert-score metric, so dropping it
card_for_test = deepcopy(card)
card_for_test.task.metrics = ["metrics.rouge"]

test_card(
    card_for_test,
    strict=True,
    demos_taken_from="test",
)
add_to_catalog(
    card, "cards.rag.response_generation.train.open_australian_legal_qa", overwrite=True
)
