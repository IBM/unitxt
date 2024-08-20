from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    TaskCard,
)
from unitxt.operators import FilterByCondition, Rename, Set
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

subset_dict = {
    "chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "chat-hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

for subset in subset_dict.keys():
    card = TaskCard(
        loader=LoadHF(path="allenai/reward-bench", split="filtered"),
        preprocess_steps=[
            RenameSplits({"filtered": "test"}),
            Rename(
                field_to_field={
                    "prompt": "question",
                    "chosen": "answer_a",
                    "rejected": "answer_b",
                    "subset": "group",
                }
            ),
            Set(fields={"winner": "choice_a"}),
            FilterByCondition(values={"group": subset_dict[subset]}, condition="in"),
        ],
        task="tasks.response_assessment.pairwise_comparison.single_turn",
        templates=[
            "templates.response_assessment.pairwise_comparison.mt_bench_single_turn"
        ],
    )

    test_card(card, demos_taken_from="test", strict=False, loader_limit=10000)
    subset_label = subset.replace("-", "_")
    add_to_catalog(card, f"cards.reward_bench.{subset_label}", overwrite=True)
