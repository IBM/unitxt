from unitxt import add_to_catalog
from unitxt.blocks import ToString, ToStringStripped
from unitxt.operator import SequentialOperator
from unitxt.settings_utils import get_constants

constants = get_constants()

add_to_catalog(
    SequentialOperator(
        steps=[
            ToString(field="prediction", process_every_value=False),
            ToString(
                field="references",
                process_every_value=True,
                dont_apply_to_streams=[constants.inference_stream],
            ),
        ]
    ),
    "processors.to_string",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            ToStringStripped(field="prediction", process_every_value=False),
            ToStringStripped(
                field="references",
                process_every_value=True,
                dont_apply_to_streams=[constants.inference_stream],
            ),
        ]
    ),
    "processors.to_string_stripped",
    overwrite=True,
)
