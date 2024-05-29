from unitxt import add_to_catalog
from unitxt.operator import (
    SequentialOperator,
)
from unitxt.processors import ExtractSafeUnsafeJudgment

add_to_catalog(
    SequentialOperator(
        steps=[
            ExtractSafeUnsafeJudgment(field="prediction", process_every_value=False),
            ExtractSafeUnsafeJudgment(field="references", process_every_value=True),
        ]
    ),
    "processors.safe_unsafe",
    overwrite=True,
)
