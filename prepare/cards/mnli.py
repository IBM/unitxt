from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    NormalizeListFields,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card
from src.unitxt.splitters import RenameSplits
from src.unitxt import dataset
import datasets as ds


card = TaskCard(
    loader=LoadHF(path="glue", name="mnli"),
    preprocess_steps=[
        RenameSplits({"validation_matched": "validation"}),
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}),
        AddFields(
            fields={
                "choices": ["entailment", "neutral", "contradiction"],
            }
        ),
    ],
    task='tasks.nli',
    templates='templates.nli',
)

test_card(card)
add_to_catalog(card, "cards.mnli", overwrite=True)

mnli_dataset = ds.load_dataset("/u/shachardon/repo/unitxt/src/unitxt/dataset.py", "card=cards.mnli")
print()