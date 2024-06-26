from unitxt import add_to_catalog
from unitxt.operator import (
    SequentialOperator,
)
from unitxt.processors import ExtractSafeUnsafeJudgment
from unitxt.settings_utils import get_constants

constants = get_constants()

add_to_catalog(
    SequentialOperator(
        steps=[
            ExtractSafeUnsafeJudgment(field="prediction", process_every_value=False),
            ExtractSafeUnsafeJudgment(
                field="references",
                process_every_value=True,
                dont_apply_to_streams=[constants.inference_stream],
            ),
        ]
    ),
    "processors.safe_unsafe",
    overwrite=True,
)
