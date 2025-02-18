from unitxt import add_to_catalog
from unitxt.operator import SequentialOperator
from unitxt.processors import AddPrefix, GetSQL

add_to_catalog(
    SequentialOperator(
        steps=[
            AddPrefix(field="prediction", prefix="SELECT "),
            GetSQL(field="prediction"),
        ]
    ),
    "processors.text2sql.get_sql",
    overwrite=True,
)
