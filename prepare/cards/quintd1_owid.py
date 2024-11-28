from unitxt.blocks import (
    LoadHF,
    SerializeTableAsMarkdown,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import RenameFields
from unitxt.splitters import SplitRandomMix
from unitxt.struct_data_operators import MapTableListsToStdTableJSON
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="Shir123/quintd1_owid",
        streaming=False,
        data_classification_policy=["public"],
    ),
    # TODO: change the path after loading the script to ibm/
    preprocess_steps=[
        SplitRandomMix(
            {"train": "validation", "validation": "validation", "test": "test"}
        ),
        Set(
            fields={
                "type_of_input_a": "table",
                "type_of_input_b": "metadata",
                "type_of_output": "caption",
                "output": "caption",  # TODO: remove after solving metric issues
            }
        ),
        MapTableListsToStdTableJSON(field="table", to_field="table_json"),
        SerializeTableAsMarkdown(field="table_json", to_field="input_a"),
        RenameFields(field="metadata", to_field="input_b"),
    ],
    task="tasks.generation.from_pair[metrics=[metrics.bleu]]",  # TODO: change metric to llm_as_judge
    templates="templates.generation.from_pair.all",
)

test_card(card, num_demos=0, demos_pool_size=5, strict=False)
add_to_catalog(card, "cards.quintd1_owid", overwrite=True)
