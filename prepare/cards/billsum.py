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
        "dataset_info_tags": [
            "task_categories:summarization",
            "annotations_creators:found",
            "language_creators:found",
            "multilinguality:monolingual",
            "size_categories:10K<n<100K",
            "source_datasets:original",
            "language:en",
            "license:cc0-1.0",
            "bills-summarization",
            "croissant",
            "arxiv:1910.00523",
            "region:us",
        ]
    },
)
test_card(
    card,
    format="formats.textual_assistant",
)
add_to_catalog(card, "cards.billsum", overwrite=True)
