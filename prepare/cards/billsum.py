from unitxt import add_to_catalog
from unitxt.blocks import Set, SplitRandomMix, TaskCard
from unitxt.loaders import LoadHF
from unitxt.operators import FilterByExpression, RenameFields
from unitxt.test_utils.card import test_card

# https://huggingface.co/datasets/billsum

n_chars_to_filter_by_list = ["max", 6000, 10000]
for n_chars_to_filter_by in n_chars_to_filter_by_list:
    card = TaskCard(
        loader=LoadHF(path="billsum"),
        preprocess_steps=[
            SplitRandomMix(
                {"train": "train[87.5%]", "validation": "train[12.5%]", "test": "test"}
            ),
            RenameFields(field_to_field={"text": "document"}),
            Set(fields={"document_type": "document"}),
        ]
        + (
            [FilterByExpression(f"len(document) <= {n_chars_to_filter_by}")]
            if n_chars_to_filter_by != "max"
            else []
        ),
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
        loader_limit=50,  # the default 30 does not suffice for stream validation to pass the filtering preprocess step
        format="formats.textual_assistant",
    )
    add_to_catalog(
        card,
        f"cards.billsum{f'_document_filtered_to_{n_chars_to_filter_by}_chars' if n_chars_to_filter_by!='max' else ''}",
        overwrite=True,
    )

# from unitxt import load_dataset

# ds = load_dataset(
#     "card=cards.billsum_document_filtered_to_10000,template=templates.summarization.abstractive.formal"
# )
# ds["test"]["source"][0]
