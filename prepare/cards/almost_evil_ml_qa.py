from unitxt.blocks import LoadHF, Rename, SplitRandomMix, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import ListFieldValues
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="0x22almostEvil/multilingual-wikihow-qa-16k"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[90%]", "validation": "train[5%]", "test": "train[5%]"}
        ),
        Rename(field_to_field={"INSTRUCTION": "question"}),
        ListFieldValues(fields=["RESPONSE"], to_field="answers"),
    ],
    task="tasks.qa.open",
    templates="templates.qa.open.all",
    __tags__={
        "flags": ["QnA", "wikihow"],
        "language": ["en", "ru", "pt", "it", "es", "fr", "de", "nl"],
        "license": "cc-by-nc-3.0",
        "region": "us",
        "size_categories": "10K<n<100K",
        "task_categories": "question-answering",
    },
    __description__=(
        "Contains Parquet of a list of instructions and WikiHow articles on different languages. See the full description (and warnings) on the dataset page: https://huggingface.co/datasets/0x22almostEvil/multilingual-wikihow-qa-16k."
    ),
)

test_card(card, debug=False)
add_to_catalog(card, "cards.almost_evil", overwrite=True)
