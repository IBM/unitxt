from unitxt import add_to_catalog
from unitxt.blocks import LoadHF, TaskCard
from unitxt.operators import Copy, IndexOf, Set
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="truthfulqa/truthful_qa",
        name="multiple_choice",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        RenameSplits({"validation": "test"}),
        Set({"_label": 1}),
        Copy(
            field_to_field={
                "mc1_targets/choices": "choices",
                "mc1_targets/labels": "labels",
            }
        ),
        IndexOf(search_in="labels", index_of="_label", to_field="answer"),
    ],
    task="tasks.qa.multiple_choice.open",
    templates=[
        "templates.qa.multiple_choice.helm",
        "templates.qa.multiple_choice.match",
    ],
    __description__="TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts.",
    __tags__={
        "languages": ["english"],
        "urls": {"arxiv": "https://arxiv.org/abs/2109.07958"},
    },
)

test_card(card, strict=False)
add_to_catalog(card, "cards.safety.truthful_qa", overwrite=True)
