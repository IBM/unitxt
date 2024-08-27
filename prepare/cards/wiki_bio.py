from unitxt.blocks import (
    ListToKeyValPairs,
    LoadHF,
    Rename,
    SerializeKeyValPairs,
    Set,
    SplitRandomMix,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wiki_bio"),
    preprocess_steps=[
        SplitRandomMix({"train": "train", "validation": "val", "test": "test"}),
        ListToKeyValPairs(
            fields=["input_text/table/column_header", "input_text/table/content"],
            to_field="kvpairs",
        ),
        SerializeKeyValPairs(field_to_field=[["kvpairs", "input"]]),
        Rename(field_to_field={"target_text": "output"}),
        Set(fields={"type_of_input": "Key-Value pairs"}),
    ],
    task="tasks.generation",
    templates="templates.generation.all",
    __tags__={
        "annotations_creators": "found",
        "arxiv": "1603.07771",
        "language": "en",
        "language_creators": "found",
        "license": "cc-by-sa-3.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "100K<n<1M",
        "source_datasets": "original",
        "task_categories": "table-to-text",
    },
    __description__=(
        "This dataset gathers 728,321 biographies from wikipedia. It aims at evaluating text generation algorithms. For each article, we provide the first paragraph and the infobox (both tokenized). For each article, we extracted the first paragraph (text), the infobox (structured data). Each infobox is encoded as a list of (field name, field value) pairs. We used Stanford CoreNLP (http://stanfordnlp.github.io/CoreNLP/) to preprocess the data, i.e. we broke the text into sentences and tokenized both the text and the field values… See the full description on the dataset page: https://huggingface.co/datasets/wiki_bio"
    ),
)

test_card(card)
add_to_catalog(card, "cards.wiki_bio", overwrite=True)
