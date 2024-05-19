from unitxt.blocks import LoadHF, MapInstanceValues, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import AddFields, ExtractFieldValues, RenameFields
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="sst2"),
    preprocess_steps=[
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "negative", "1": "positive"}}),
        RenameFields(field="sentence", to_field="text"),
        AddFields(
            fields={
                "text_type": "sentence",
                "type_of_class": "sentiment",
            }
        ),
        ExtractFieldValues(field="label", to_field="classes", stream_name="train"),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
    __tags__={
        "annotations_creators": "other",
        "arxiv": "1804.07461",
        "coreference-nli": True,
        "croissant": True,
        "language": "en",
        "language_creators": "other",
        "license": "other",
        "multilinguality": "monolingual",
        "paraphrase-identification": True,
        "qa-nli": True,
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": [
            "acceptability-classification",
            "natural-language-inference",
            "semantic-similarity-scoring",
            "sentiment-classification",
            "text-scoring",
        ],
    },
)

test_card(card, debug=True)
add_to_catalog(card, "cards.sst2", overwrite=True)
