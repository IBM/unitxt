from unitxt import add_to_catalog
from unitxt.blocks import Set, SplitRandomMix, TaskCard
from unitxt.loaders import LoadHF
from unitxt.operators import FilterByExpression, RenameFields
from unitxt.test_utils.card import test_card

n_chars_to_filter_by_list = ["max", 6000, 10000]
for n_chars_to_filter_by in n_chars_to_filter_by_list:
    card = TaskCard(
        loader=LoadHF(path="webis/tldr-17", streaming=True),
        preprocess_steps=[
            SplitRandomMix({"train": "train[50%]", "test": "train[50%]"}),
            RenameFields(field_to_field={"content": "document"}),
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
            "annotations_creators": "no-annotation",
            "flags": ["reddit-posts-summarization"],
            "language": "en",
            "language_creators": "crowdsourced",
            "license": "cc-by-4.0",
            "multilinguality": "monolingual",
            "region": "us",
            "size_categories": "1M<n<10M",
            "source_datasets": "original",
            "task_categories": "summarization",
        },
        __description__=(
            "This corpus contains preprocessed posts from the Reddit dataset.\n"
            "The dataset consists of 3,848,330 posts with an average length of 270 words for content,\n"
            "and 28 words for the summary.\n"
            "Features includes strings: author, body, normalizedBody, content, summary, subreddit, subreddit_id.\n"
            "Content is used as document and summary is used as summary."
        ),
    )
    test_card(
        card,
        format="formats.textual_assistant",
    )
    add_to_catalog(
        card,
        f"cards.tldr{f'_document_filtered_to_{n_chars_to_filter_by}_chars' if n_chars_to_filter_by!='max' else ''}",
        overwrite=True,
    )
