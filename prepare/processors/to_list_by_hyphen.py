from unitxt import add_to_catalog
from unitxt.operator import SequentialOperator
from unitxt.operators import RemoveValues
from unitxt.processors import ToListByHyphenSpace

add_to_catalog(
    SequentialOperator(
        steps=[
            ToListByHyphenSpace(field="prediction", process_every_value=False),
            RemoveValues(
                field="prediction",
                unallowed_values=["", " "],
                process_every_value=False,
            ),
            ToListByHyphenSpace(field="references", process_every_value=True),
            RemoveValues(
                field="references",
                unallowed_values=["", " "],
                process_every_value=True,
            ),
        ]
    ),
    "processors.to_list_by_hyphen_space",
    overwrite=True,
)
add_to_catalog(
    SequentialOperator(
        steps=[
            ToListByHyphenSpace(field="references", process_every_value=True),
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
