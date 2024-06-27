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
    MapInstanceValues,
    Set,
)
from unitxt.test_utils.card import test_card

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
        MapInstanceValues(
            mappers={"reference_answers": {"['']": unanswerable_responses}},
            strict=False,
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

debug = False
if debug:
    from unitxt import evaluate, load_dataset

    ds = load_dataset(
        "card=cards.rag.response_generation.clapnq,template=templates.rag.response_generation.answer_based_on_context_inverted,demos_pool_size=100,num_demos=2,format=formats.llama3_chat,system_prompt=system_prompts.empty,demos_taken_from=train,augmentor=augmentors.no_augmentation,demos_removed_from_data=True",
    )

    predictions = ds["test"]["target"]
    # 3. create a metric and evaluate the results.
    scores = evaluate(predictions=predictions, data=ds["test"])
