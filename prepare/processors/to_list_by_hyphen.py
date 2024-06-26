from unitxt import add_to_catalog
from unitxt.operator import SequentialOperator
from unitxt.operators import RemoveValues
from unitxt.settings_utils import get_constants
from unitxt.string_operators import RegexSplit

constants = get_constants()
regex = "(?:^|\n)- "

add_to_catalog(
    SequentialOperator(
        steps=[
            RegexSplit(field="prediction", by=regex),
            RemoveValues(
                field="prediction",
                unallowed_values=["", " "],
                process_every_value=False,
            ),
            RegexSplit(field="references", by=regex, process_every_value=True),
            RemoveValues(
                field="references",
                unallowed_values=["", " "],
                process_every_value=True,
                dont_apply_to_streams=[constants.inference_stream],
            ),
        ]
    ),
    "processors.to_list_by_hyphen_space",
    overwrite=True,
)
add_to_catalog(
    SequentialOperator(
        steps=[
            RegexSplit(field="references", by=regex, process_every_value=True),
            RemoveValues(
                field="references",
                unallowed_values=["", " "],
                process_every_value=True,
            ),
        ]
    ),
    "processors.to_list_by_hyphen_space_from_references",
    overwrite=True,
)
