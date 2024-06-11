from unitxt.blocks import (
    CopyFields,
    LoadHF,
    RenameFields,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="openai/summarize_from_feedback", name="comparisons"),
    preprocess_steps=[
        "splitters.small_no_test",
        CopyFields(
            field_to_field={
                "info/post": "input",
                "summaries/*/text": "choices",
            }
        ),
        RenameFields(field_to_field={"choice": "output_choice"}),
        Set(
            fields={
                "input_type": "post",
                "output_type": "summary",
                "instruction": "Summarize the following post",
            }
        ),
    ],
    task="tasks.evaluation.preference",
    templates="templates.evaluation.preference.all",
    __tags__={"arxiv": "2009.01325", "region": "us"},
    __description__=(
        'Summarize from Feedback contains the human feedback data released by the "Learning to summarize from human feedback" paper… See the full description on the dataset page: https://huggingface.co/datasets/openai/summarize_from_feedback'
    ),
)

test_card(card, strict=False)
add_to_catalog(card, "cards.summarize_from_human_feedback", overwrite=True)
