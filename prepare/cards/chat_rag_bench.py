from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
)
from unitxt.dialog_operators import SerializeOpenAiFormatDialog
from unitxt.operators import Copy, Set, Shuffle
from unitxt.test_utils.card import test_card

splits_random_mixes = {
    "train": SplitRandomMix(
        {"train": "test[0.6]", "validation": "test[0.2]", "test": "test[0.2]"}
    ),
    "standard": SplitRandomMix({"test": "test"}),
}

subsets = ["doqa_travel", "doqa_cooking", "doqa_movies", "doc2dial", "hybridial"]
for split in splits_random_mixes:
    for subset in subsets:
        card = TaskCard(
            loader=LoadHF(
                path="nvidia/ChatRAG-Bench",
                name=subset,
                split="test",
                all_splits=["test"],
            ),
            preprocess_steps=[
                splits_random_mixes[split],
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
            templates={
                "default": "templates.rag.response_generation.please_respond_chat"
            },
        )

        if subset == "doqa_travel":
            test_card(
                card,
                strict=True,
                demos_taken_from="test",
                metrics=[
                    "metrics.rag.response_generation.answer_correctness.token_recall",
                    "metrics.rag.response_generation.faithfulness.token_k_precision",
                    "metrics.rag.response_generation.answer_relevance.token_recall",
                ],
            )
        add_to_catalog(
            card,
            f"cards.rag.response_generation.chat_rag_bench.{'train.' if split == 'train' else ''}user_assistant_format.{subset}",
            overwrite=True,
        )
