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
    Set,
)
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="PrimeQA/clapnq",
    ),
    preprocess_steps=[
        SplitRandomMix({"train": "train", "test": "validation"}),
        Copy(
            field_to_field={
                "passages/*/text": "contexts",
                "input": "question",
                "output/*/answer": "reference_answers",
            }
        ),
        Set(
            fields={
                "contexts_ids": [],
            }
        ),
    ],
    task="tasks.rag.response_generation",
    templates=TemplatesDict(
        {
            "please_respond": "templates.rag.response_generation.please_respond",
            "answer_based_on_context": "templates.rag.response_generation.answer_based_on_context",
            "answer_based_on_context_inverted": "templates.rag.response_generation.answer_based_on_context_inverted",
        }
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
add_to_catalog(
    card,
    "cards.rag.response_generation.clapnq",
    overwrite=True,
)
