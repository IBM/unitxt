from unitxt import add_to_catalog
from unitxt.operator import SequentialOperator
from unitxt.text2sql.processors import GetSQL

add_to_catalog(
    SequentialOperator(
        steps=[
            GetSQL(field="prediction"),
            # AddPrefix(field="prediction""),
        ]
    ),
    "processors.text2sql.get_sql",
    overwrite=True,
)
