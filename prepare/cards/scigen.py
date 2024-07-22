from unitxt.blocks import (
    ConstructTableStructure,
    LoadHF,
    RenameFields,
    SerializeTableAsIndexedRowMajor,
    Set,
    SplitRandomMix,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="kasnerz/scigen"),
    preprocess_steps=[
        SplitRandomMix({"train": "train", "validation": "validation", "test": "test"}),
        ConstructTableStructure(
            fields=["table_column_names", "table_content_values", "table_caption"],
            to_field="table",
        ),
        SerializeTableAsIndexedRowMajor(field_to_field=[["table", "input"]]),
        Set({"type_of_input": "Table"}),
        Set({"type_of_output": "Text"}),
        RenameFields(field_to_field={"text": "output"}),
    ],
    task="tasks.generation[metrics=[metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_table2text_single_turn_with_reference]]",
    templates="templates.generation.all",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.scigen", overwrite=True)
