from unitxt.blocks import LoadHF, RenameFields, SplitRandomMix, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import ListFieldValues
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="0x22almostEvil/multilingual-wikihow-qa-16k"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[90%]", "validation": "train[5%]", "test": "train[5%]"}
        ),
        RenameFields(field_to_field={"INSTRUCTION": "question"}),
        ListFieldValues(fields=["RESPONSE"], to_field="answers"),
    ],
    task="tasks.qa.open",
    templates="templates.qa.open.all",
    __tags__={
        "flags": ["QnA", "croissant", "wikihow"],
        "language": ["en", "ru", "pt", "it", "es", "fr", "de", "nl"],
        "license": "cc-by-nc-3.0",
        "region": "us",
        "size_categories": "10K<n<100K",
        "task_categories": "question-answering",
    },
    __description__=(
        "Dataset Card for multilingual WikiHow with ~16.8K entries. ~(2-2.2)K for each language.\n"
        "Warning [1]\n"
        "The WikiHow team contacted me and made it clear that they forbid the use of their data for machine learning purposes. However, I am not calling for anything, and this dataset only shows the concept, and I strongly advise against violating their ToS.\n"
        "However, consultation with lawyers made it clear that dataset can be used for such purposes if the project has… See the full description on the dataset page: https://huggingface.co/datasets/0x22almostEvil/multilingual-wikihow-qa-16k."
    ),
)

test_card(card, debug=False)
add_to_catalog(card, "cards.almost_evil", overwrite=True)
