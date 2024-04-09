from unitxt import add_to_catalog
from unitxt.operator import SequentialOperator
from unitxt.processors import ToListByComma

add_to_catalog(
    SequentialOperator(
        steps=[
            ToListByComma(field="prediction", process_every_value=False),
            ToListByComma(field="references", process_every_value=True),
        ]
    ),
    "processors.to_list_by_comma",
    overwrite=True,
)
