from unitxt.blocks import LoadHF, MapInstanceValues, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import ExtractFieldValues, Rename, Set
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="sst2"),
    preprocess_steps=[
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "negative", "1": "positive"}}),
        Rename(field="sentence", to_field="text"),
        Set(
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
        "flags": ["coreference-nli", "paraphrase-identification", "qa-nli"],
        "language": "en",
        "language_creators": "other",
        "license": "other",
        "multilinguality": "monolingual",
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
    __description__=(
        "The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. It uses the two-way (positive/negative) class split, with only sentence-level labels… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
)

test_card(card)
add_to_catalog(card, "cards.sst2", overwrite=True)
