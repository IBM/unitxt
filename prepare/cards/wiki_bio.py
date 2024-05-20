from unitxt.blocks import (
    AddFields,
    ListToKeyValPairs,
    LoadHF,
    RenameFields,
    SerializeKeyValPairs,
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
        RenameFields(field_to_field={"target_text": "output"}),
        AddFields(
            fields={"type_of_input": "Key-Value pairs", "type_of_output": "Text"}
        ),
    ],
    task="tasks.generation",
    templates="templates.generation.all",
    __tags__={
        "annotations_creators": "found",
        "arxiv": "1603.07771",
        "croissant": True,
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
        "This dataset gathers 728,321 biographies from wikipedia. It aims at evaluating text generation\n"
        "algorithms. For each article, we provide the first paragraph and the infobox (both tokenized).\n"
        "For each article, we extracted the first paragraph (text), the infobox (structured data). Each\n"
        "infobox is encoded as a list of (field name, field value) pairs. We used Stanford CoreNLP\n"
        "(http://stanfordnlp.github.io/CoreNLP/) to preprocess the data, i.e. we broke the text into\n"
        "sentences and tokenized both the text and the field values. The dataset was randomly split in\n"
        "three subsets train (80%), valid (10%), test (10%)."
    ),
)

test_card(card)
add_to_catalog(card, "cards.wiki_bio", overwrite=True)
