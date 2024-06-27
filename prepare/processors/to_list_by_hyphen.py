from unitxt import add_to_catalog
from unitxt.operator import SequentialOperator
from unitxt.operators import RemoveValues
from unitxt.processors import PostProcess
from unitxt.settings_utils import get_constants
from unitxt.string_operators import RegexSplit

constants = get_constants()
regex = "(?:^|\n)- "

add_to_catalog(
    SequentialOperator(
        steps=[
            PostProcess(RegexSplit(by=regex)),
            PostProcess(RemoveValues(unallowed_values=["", " "])),
        ]
    ),
    "processors.to_list_by_hyphen_space",
    overwrite=True,
)
add_to_catalog(
    SequentialOperator(
        steps=[
            PostProcess(RegexSplit(by=regex), process_prediction=False),
            PostProcess(
                RemoveValues(unallowed_values=["", " "]), process_prediction=False
            ),
        ]
    ),
    "processors.to_list_by_hyphen_space_from_references",
    overwrite=True,
)
