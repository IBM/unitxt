from typing import Any, Dict, List

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        __description__="This is a key value extraction task, where a specific list of possible 'keys' need to be extracted from the input.  The ground truth is provided key-value pairs in the form of the dictionary.  The results are evaluating using F1 score metric, that expects the predictions to be converted into a list of (key,value) pairs. ",
        input_fields={"input": Any, "keys": List[str]},
        reference_fields={"key_value_pairs_answer": Dict[str, str]},
        prediction_type=Dict[str, str],
        metrics=[
            "metrics.key_value_extraction.accuracy",
            "metrics.key_value_extraction.token_overlap",
        ],
        default_template="templates.key_value_extraction.extract_in_json_format",
    ),
    "tasks.key_value_extraction",
    overwrite=True,
)
