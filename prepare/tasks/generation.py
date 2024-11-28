from typing import Union

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog
from unitxt.types import Audio, Dialog, Image, Table, Text

add_to_catalog(
    Task(
        input_fields={"input": str, "type_of_input": str, "type_of_output": str},
        reference_fields={"output": str},
        prediction_type=str,
        metrics=["metrics.normalized_sacrebleu"],
        augmentable_inputs=["input"],
        defaults={"type_of_output": "Text"},
    ),
    "tasks.generation",
    overwrite=True,
)

add_to_catalog(
    Task(
        input_fields={
            "input_a": Union[Text, Image, Audio, Table, Dialog],
            "type_of_input_a": str,
            "input_b": Union[Text, Image, Audio, Table, Dialog],
            "type_of_input_b": str,
            "type_of_output": str,
        },
        reference_fields={"output": str},
        prediction_type=str,
        metrics=[
            "metrics.bleu",
            "metrics.rouge",
            "metrics.bert_score.bert_base_uncased",
            "metrics.meteor",
        ],
        augmentable_inputs=["input_a", "input_b"],
        defaults={"type_of_output": "Text"},
    ),
    "tasks.generation.from_pair",
    overwrite=True,
)
