from unitxt import add_to_catalog
from unitxt.blocks import (
    FormTask,
    InputOutputTemplate,
    LoadHF,
    TaskCard,
    TemplatesList,
)
from unitxt.operators import (
    AddFields,
    CopyFields,
    FilterByCondition,
    Shuffle,
)
from unitxt.splitters import RenameSplits
from unitxt.struct_data_operators import DumpJson
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="AlexaAI/bold"),
    preprocess_steps=[
        RenameSplits(mapper={"train": "test"}),
        AddFields({"input_label": {}}),
        CopyFields(field_to_field=[("prompts/0", "first_prompt")]),
        CopyFields(field_to_field=[("wikipedia/0", "first_wiki")]),
        FilterByCondition(values={"domain": ["race", "gender"]}, condition="in"),
        Shuffle(page_size=10000),
        CopyFields(
            field_to_field={
                "first_prompt": "input_label/input",
                "category": "input_label/category",
                "first_wiki": "input_label/wiki",
            },
        ),
        DumpJson(field="input_label"),
    ],
    task=FormTask(
        inputs=["first_prompt"], outputs=["input_label"], metrics=["metrics.regard"]
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="{first_prompt}\n", output_format="{input_label}"
            ),
        ]
    ),
)

test_card(card, demos_taken_from="test", strict=False)
add_to_catalog(card, "cards.bold", overwrite=True)
