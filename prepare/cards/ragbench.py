from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    TaskCard,
)
from unitxt.collections_operators import Dictify, Wrap
from unitxt.operators import Copy, Set
from unitxt.test_utils.card import test_card

for subset in [
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
]:
    card = TaskCard(
        loader=LoadHF(
            path="rungalileo/ragbench",
            name=subset,
        ),
        preprocess_steps=[
            Copy(field="documents", to_field="contexts"),
            Copy(field="documents", to_field="contexts_ids"),
            Wrap(field="response", inside="list", to_field="reference_answers"),
        ],
        task="tasks.rag.response_generation",
        templates={"default": "templates.rag.response_generation.please_respond_chat"},
    )

    if subset == "covidqa":
        test_card(
            card,
            strict=True,
            metrics=[
                "metrics.rag.response_generation.answer_correctness.token_recall",
                "metrics.rag.response_generation.faithfulness.token_k_precision",
                "metrics.rag.response_generation.answer_relevance.token_recall",
            ],
            demos_taken_from="test",
        )

    add_to_catalog(
        card, f"cards.rag.response_generation.ragbench.{subset}", overwrite=True
    )

    card = TaskCard(
        loader=LoadHF(
            path="rungalileo/ragbench",
            name=subset,
        ),
        preprocess_steps=[
            Set({"context_type": "documents"}),
            Wrap(field="documents", inside="list", process_every_value=True),
            Dictify(
                field="documents",
                to_field="context",
                with_keys=["body"],
                process_every_value=True,
            ),
            Set({"context/*/title": "Document"}),
            Wrap(field="response", inside="list", to_field="answers"),
        ],
        task="tasks.qa.with_context",
        templates="templates.qa.with_context.all",
    )

    if subset == "covidqa":
        test_card(
            card,
            strict=True,
            demos_taken_from="test",
        )

    add_to_catalog(card, f"cards.ragbench.{subset}", overwrite=True)
