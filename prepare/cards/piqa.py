from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import ListFieldValues, RenameFields
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="piqa"),
    preprocess_steps=[
        ListFieldValues(fields=["sol1", "sol2"], to_field="choices"),
        RenameFields(
            field_to_field={"goal": "question", "label": "answer"},
        ),
    ],
    task="tasks.qa.multiple_choice.open",
    templates="templates.qa.multiple_choice.open.all",
    __tags__={
        "annotations_creators": "crowdsourced",
        "arxiv": ["1911.11641", "1907.10641", "1904.09728", "1808.05326"],
        "language": "en",
        "language_creators": ["crowdsourced", "found"],
        "license": "unknown",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "question-answering",
        "task_ids": "multiple-choice-qa",
    },
    __description__=(
        "To apply eyeshadow without a brush, should I use a cotton swab or a toothpick? Questions requiring this kind of physical commonsense pose a challenge to state-of-the-art natural language understanding systems. The PIQA dataset introduces the task of physical commonsense reasoning and a corresponding benchmark dataset Physical Interaction: Question Answering or PIQA. Physical commonsense knowledge is a major challenge on the road to true AI-completeness… See the full description on the dataset page: https://huggingface.co/datasets/piqa"
    ),
)
test_card(card, strict=False)
add_to_catalog(card, "cards.piqa", overwrite=True)
