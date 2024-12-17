from unitxt import add_to_catalog
from unitxt.blocks import (
    ConstructTableFromRowsCols,
    LoadHF,
    Rename,
    Set,
    TaskCard,
)
from unitxt.operators import FilterByCondition
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="kasnerz/scigen", data_classification_policy=["public", "proprietary"]
    ),
    preprocess_steps=[
        FilterByCondition(values={"table_content_values": "[]"}, condition="ne"),
        ConstructTableFromRowsCols(
            fields=["table_column_names", "table_content_values"],
            to_field="input_a",
        ),
        Rename(field_to_field={"table_caption": "input_b", "text": "output"}),
        Set(
            fields={
                "type_of_input_a": "table",
                "type_of_input_b": "caption",
                "type_of_output": "text description",
            }
        ),
    ],
    task="tasks.generation.from_pair[metrics=[metrics.llm_as_judge.rating.llama_3_1_70b_instruct_cross_provider_template_table2text_single_turn_with_reference]]",
    templates=[
        "templates.generation.from_pair.default[postprocessors=[processors.lower_case]]"
    ],
    __description__="SciGen is a dataset for the task of reasoning-aware data-to-text generation. It consists of tables from scientific articles(mostly containing numerical values) and their corresponding text descriptions.",
    __tags__={
        "modality": "table",
        "urls": {"arxiv": "https://arxiv.org/pdf/2104.08296"},
        "languages": ["english"],
    },
)

test_card(card, strict=False)
add_to_catalog(card, "cards.scigen", overwrite=True)
