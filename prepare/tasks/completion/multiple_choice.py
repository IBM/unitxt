from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"context": "str", "context_type": "str", "choices": "List[str]"},
        outputs={"answer": "int", "choices": "List[str]"},
        prediction_type="Any",
        metrics=["metrics.accuracy"],
    ),
    "tasks.completion.multiple_choice",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs={"context": "str", "context_type": "str", "completion_type": "str"},
        outputs={"completion": "str"},
        prediction_type="str",
        metrics=["metrics.rouge"],
    ),
    "tasks.completion.abstractive",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs={"context": "str", "context_type": "str", "completion_type": "str"},
        outputs={"completion": "str"},
        prediction_type="Dict[str,Any]",
        metrics=["metrics.squad"],
    ),
    "tasks.completion.extractive",
    overwrite=True,
)
