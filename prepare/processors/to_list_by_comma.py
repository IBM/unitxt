from unitxt import add_to_catalog
from unitxt.operator import SequentialOperator
from unitxt.processors import ToListByComma
from unitxt.settings_utils import get_constants

constants = get_constants()
add_to_catalog(
    SequentialOperator(
        steps=[
            ToListByComma(field="prediction", process_every_value=False),
            ToListByComma(
                field="references",
                process_every_value=True,
                dont_apply_to_streams=[constants.inference_stream],
            ),
        ]
    ),
    "processors.to_list_by_comma",
    overwrite=True,
)

add_to_catalog(
    ToListByComma(field="references", process_every_value=True),
    "processors.to_list_by_comma_from_references",
    overwrite=True,
)
