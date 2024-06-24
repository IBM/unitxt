from unitxt import add_to_catalog
from unitxt.blocks import SplitRandomMix, TaskCard
from unitxt.loaders import LoadHF
from unitxt.operators import RenameFields
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="webis/tldr-17", streaming=True),
    preprocess_steps=[
        SplitRandomMix({"train": "train[50%]", "test": "train[50%]"}),
        RenameFields(field_to_field={"content": "document"}),
    ],
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
add_to_catalog(card, "cards.tldr", overwrite=True)
