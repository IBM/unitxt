from unitxt.blocks import (
    ConstructTableFromRowsCols,
    LoadHF,
    RenameFields,
    SerializeTableAsIndexedRowMajor,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import FilterByCondition
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate, TemplatesList
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="kasnerz/scigen", data_classification_policy=["public"]),
    preprocess_steps=[
        FilterByCondition(values={"table_content_values": "[]"}, condition="ne"),
        ConstructTableFromRowsCols(
            fields=["table_column_names", "table_content_values"],
            to_field="table",
        ),
        SerializeTableAsIndexedRowMajor(field_to_field=[["table", "input"]]),
        Set({"type_of_input": "table"}),
        Set({"type_of_output": "text"}),
        RenameFields(field_to_field={"text": "output", "table_caption": "caption"}),
    ],
    task=Task(
        input_fields={
            "input": "str",
            "type_of_input": "str",
            "caption": "str",
            "type_of_output": "str",
        },
        reference_fields={"output": "str"},
        prediction_type="str",
        metrics=[
            "metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_table2text_single_turn_with_reference"
        ],
        augmentable_inputs=["input"],
        defaults={"type_of_output": "Text"},
    ),
    templates=TemplatesList(
        [  # TODO: set to "templates.generation.structured" after numeric nlg PR is approved
            InputOutputTemplate(
                input_format="Given the following {type_of_input} and caption, generate the corresponding {type_of_output}."
                "\n{type_of_input}: \n{input} \ncaption: \n{caption} \n{type_of_output}:",
                output_format="{output}",
                postprocessors=[
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                ],
            ),
        ]
    ),
)

test_card(card, strict=False)
add_to_catalog(card, "cards.scigen", overwrite=True)
