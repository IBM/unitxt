from unitxt import add_to_catalog
from unitxt.blocks import AddFields, TaskCard
from unitxt.loaders import LoadHF
from unitxt.operators import RenameFields
from unitxt.test_utils.card import test_card

# https://huggingface.co/datasets/billsum
card = TaskCard(
    loader=LoadHF(path="billsum"),
    preprocess_steps=[
        RenameFields(field_to_field={"text": "document"}),
        AddFields(fields={"document_type": "document"}),
    ],
    task="tasks.summarization.abstractive",
    templates="templates.summarization.abstractive.all",
    __tags__={
        "annotations_creators": "found",
        "arxiv": "1910.00523",
        "bills-summarization": True,
        "croissant": True,
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
        'Dataset Card for "billsum"\n'
        "Dataset Summary\n"
        "BillSum, summarization of US Congressional and California state bills.\n"
        "There are several features:\n"
        "text: bill text.\n"
        "summary: summary of the bills.\n"
        "title: title of the bills.\n"
        "features for us bills. ca bills does not have.\n"
        "text_len: number of chars in text.\n"
        "sum_len: number of chars in summary.\n"
        "Supported Tasks and Leaderboards\n"
        "More Information Needed\n"
        "Languages\n"
        "More Information Needed… See the full description on the dataset page: https://huggingface.co/datasets/billsum."
    ),
)
test_card(
    card,
    format="formats.textual_assistant",
)
add_to_catalog(card, "cards.billsum", overwrite=True)
