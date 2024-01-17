from src.unitxt import add_to_catalog
from src.unitxt.blocks import ToString, ToStringStripped
from src.unitxt.operator import SequentialOperator

add_to_catalog(
    SequentialOperator(
        steps=[
            ToString(field="prediction", process_every_value=False),
            ToString(field="references", process_every_value=True),
        ]
    ),
    "processors.to_string",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            ToStringStripped(field="prediction", process_every_value=False),
            ToStringStripped(field="references", process_every_value=True),
        ]
    ),
    "processors.to_string_stripped",
    overwrite=True,
)
