from unitxt import add_to_catalog
from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    Task,
    TaskCard,
    TemplatesList,
)
from unitxt.operators import Copy, Set, Shuffle
from unitxt.splitters import RenameSplits
from unitxt.struct_data_operators import DumpJson
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="ibm/AttaQ"),
    preprocess_steps=[
        RenameSplits(mapper={"train": "test"}),
        Shuffle(page_size=2800),
        Set({"input_label": {}}),
        Copy(
            field_to_field={"input": "input_label/input", "label": "input_label/label"}
        ),
        DumpJson(field="input_label"),
    ],
    task=Task(
        input_fields=["input"],
        reference_fields=["input_label"],
        metrics=["metrics.safety_metric"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="{input}\n", output_format="{input_label}"
            ),
            InputOutputTemplate(input_format="{input}", output_format="{input_label}"),
        ]
    ),
)

test_card(card, strict=False, demos_taken_from="test", num_demos=0)
add_to_catalog(card, "cards.atta_q", overwrite=True)
