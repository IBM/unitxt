from unitxt import add_to_catalog
from unitxt.blocks import TaskCard
from unitxt.loaders import LoadHF
from unitxt.operators import RenameFields
from unitxt.splitters import SplitRandomMix
from unitxt.test_utils.card import test_card

# https://huggingface.co/datasets/billsum
card = TaskCard(
    loader=LoadHF(path="billsum"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[87.5%]", "validation": "train[12.5%]", "test": "test"}
        ),
        RenameFields(field_to_field={"text": "document"}),
    ],
    task="tasks.summarization.abstractive",
    templates="templates.summarization.abstractive.all",
    __tags__={
        "annotations_creators": "found",
        "arxiv": "1910.00523",
        "flags": ["bills-summarization"],
        "language": "en",
        "language_creators": "found",
        "license": "cc0-1.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "summarization",
    },
    __description__=(
        "BillSum, summarization of US Congressional and California state bills… See the full description on the dataset page: https://huggingface.co/datasets/billsum."
    ),
)
test_card(
    card,
    format="formats.textual_assistant",
)
add_to_catalog(card, "cards.billsum", overwrite=True)
