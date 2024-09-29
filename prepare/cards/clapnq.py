from copy import deepcopy

from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
)
from unitxt.operators import (
    Copy,
    MapInstanceValues,
    Set,
)
from unitxt.test_utils.card import test_card

splits = {
    "eval": {"train": "train", "test": "validation"},
    "train": {"train": "train[0.5]", "validation": "train[0.5]", "test": "validation"},
}

unanswerable_responses = [
    "I'm sorry, I cannot answer this question based on the context.",
    "The answer is not in the text provided.",
    "Unanswerable.",
    "The provided context does not contain the information needed to answer this question.",
    "There is not enough information in the text to answer this question.",
    "The text does not provide an answer to this question.",
    "Based on the context, an answer cannot be determined.",
    "The answer to this question is not available in the provided context.",
    "This question cannot be answered with the given information.",
    "Insufficient context to provide an answer.",
]

for split in splits.keys():
    card = TaskCard(
        loader=LoadHF(
            path="PrimeQA/clapnq",
        ),
        preprocess_steps=[
            SplitRandomMix(splits[split]),
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
            MapInstanceValues(
                mappers={"reference_answers": {"['']": unanswerable_responses}},
                strict=False,
            ),
        ],
        task="tasks.rag.response_generation",
        templates={
            "please_respond": "templates.rag.response_generation.please_respond",
            "answer_based_on_context": "templates.rag.response_generation.answer_based_on_context",
            "answer_based_on_context_inverted": "templates.rag.response_generation.answer_based_on_context_inverted",
        },
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
        f'cards.rag.response_generation.{"train." if split == "train" else ""}clapnq',
        overwrite=True,
    )
