from unitxt import add_to_catalog
from unitxt.processors import PostProcess
from unitxt.struct_data_operators import (
    JsonStrToListOfKeyValuePairs,
    LiteralStrToListOfKeyValuePairs,
)
from unitxt.templates import (
    InputOutputTemplate,
)

add_to_catalog(
    InputOutputTemplate(
        instruction="Extract the key value pairs from the input. Return a valid json object with the following keys: {keys}. Return only the json representation, no additional text or explanations.",
        input_format="{input}",
        output_format="{key_value_pairs_answer}",
        postprocessors=[
            PostProcess(JsonStrToListOfKeyValuePairs(), process_references=False),
            PostProcess(LiteralStrToListOfKeyValuePairs(), process_prediction=False),
        ],
    ),
    "templates.key_value_extraction.extract_in_json_format",
    overwrite=True,
)
