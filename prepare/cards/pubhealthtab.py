import os

from unitxt.blocks import (
    LoadHF,
    RenameFields,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

data_path = "https://raw.githubusercontent.com/mubasharaak/PubHealthTab/main/data/"

data_files = {
    "train": os.path.join(data_path, "pubhealthtab_trainset.jsonl"),
    "validation": os.path.join(data_path, "pubhealthtab_evalset.jsonl"),
    "test": os.path.join(data_path, "pubhealthtab_testset.jsonl"),
}

card = TaskCard(
    loader=LoadHF(path="json", data_files=data_files),
    preprocess_steps=[
        # "splitters.small_no_test",
        # MapHTMLTableToJSON(field_to_field=[["table/html_code", "stdtable"]]),
        # SerializeTableAsIndexedRowMajor(
        #     field_to_field=[["stdtable", "table_serialized"]]
        # ),
        RenameFields(
            field_to_field={"table/html_code": "text_a", "statement": "text_b"}
        ),
        # MapInstanceValues(mappers={"label": {"0": "refuted", "1": "entailed"}}),
        Set(
            fields={
                "type_of_relation": "entailment",
                "text_a_type": "Table",
                "text_b_type": "Statement",
                "classes": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
            }
        ),
    ],
    task="tasks.classification.multi_class.relation",
    templates="templates.classification.multi_class.relation.all",
)

test_card(card)
add_to_catalog(card, "cards.pubhealthtab", overwrite=True)
