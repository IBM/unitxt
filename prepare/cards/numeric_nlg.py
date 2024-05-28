from unitxt.blocks import (
    AddFields,
    HTMLTableToJSON,
    LoadHF,
    RenameFields,
    SerializeTableAsIndexedRowMajor,
    Task,
    TaskCard,
    TemplatesList,
)
from unitxt.catalog import add_to_catalog
from unitxt.splitters import SplitRandomMix
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="kasnerz/numericnlg"),  # TODO: load from github repo
    preprocess_steps=[
        SplitRandomMix({"train": "train", "validation": "validation", "test": "test"}),
        AddFields(fields={"type_of_input": "table", "type_of_output": "text"}),
        HTMLTableToJSON(
            field_to_field=[["table_html_clean", "table_out"]]
        ),  # TODO: add tests to test_ file
        SerializeTableAsIndexedRowMajor(field_to_field=[["table_out", "input"]]),
        RenameFields(field_to_field={"description": "output"}),
    ],
    task=Task(
        inputs={"input": "str", "type_of_input": "str", "type_of_output": "str"},
        outputs={"output": "str"},
        prediction_type="str",
        metrics=["metrics.meteor"],
        augmentable_inputs=["input"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="Given the following {type_of_input}, generate the corresponding {type_of_output}. {type_of_input}: {input}",
                output_format="{output}",
                postprocessors=[],
            ),
        ]
    ),
    __description__="NumericNLG is a dataset for numerical table-to-text generation using pairs of a table and a paragraph of a table description with richer inference from scientific papers.",
    __tags__={
        "modality": "table",
        "urls": {"arxiv": "https://aclanthology.org/2021.acl-long.115/"},
        "languages": ["english"],
    },
)

test_card(card, num_demos=2)
add_to_catalog(card, "cards.numeric_nlg", overwrite=True)
