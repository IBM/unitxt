from unitxt.blocks import (
    LoadHF,
    MapHTMLTableToJSON,
    RenameFields,
    SerializeTableAsMarkdown,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="kasnerz/numericnlg"),  # TODO: load from github repo
    preprocess_steps=[
        Set(
            fields={
                "type_of_input_a": "table",
                "type_of_input_b": "caption",
                "type_of_output": "description",
            }
        ),
        MapHTMLTableToJSON(field="table_html_clean", to_field="table_out"),
        SerializeTableAsMarkdown(field="table_out", to_field="input_a"),
        RenameFields(field="description", to_field="output"),
        RenameFields(field="caption", to_field="input_b"),
    ],
    task="tasks.generation.from_pair",
    templates="templates.generation.from_pair.all",
    __description__="NumericNLG is a dataset for numerical table-to-text generation using pairs of a table and a paragraph of a table description with richer inference from scientific papers.",
    __tags__={
        "modality": "table",
        "urls": {"arxiv": "https://aclanthology.org/2021.acl-long.115/"},
        "languages": ["english"],
    },
)

test_card(card, num_demos=2, demos_pool_size=5, strict=False)
add_to_catalog(card, "cards.numeric_nlg", overwrite=True)
