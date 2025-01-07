from unitxt import add_to_catalog
from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    Task,
    TaskCard,
)
from unitxt.catalog import add_link_to_catalog
from unitxt.operators import Shuffle
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="ibm/AttaQ"),
    preprocess_steps=[
        RenameSplits(mapper={"train": "test"}),
        Shuffle(page_size=2800),
    ],
    task=Task(
        input_fields=["input"],
        reference_fields=["label"],
        metrics=["metrics.safety_metric"],
    ),
    templates=[
        InputOutputTemplate(input_format="{input}\n", output_format=""),
        InputOutputTemplate(input_format="{input}", output_format=""),
    ],
)

test_card(card, strict=False, demos_taken_from="test", num_demos=0)
add_to_catalog(card, "cards.attaq", overwrite=True)

add_link_to_catalog(
    artifact_linked_to="cards.attaq",
    name="cards.atta_q",
    overwrite=True,
)
