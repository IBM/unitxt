from unitxt import add_to_catalog
from unitxt.operator import SequentialOperator
from unitxt.text2sql.processors import AddPrefix, GetSQL, StripCodeBlock

add_to_catalog(
    SequentialOperator(
        steps=[
            StripCodeBlock(field="prediction"),
            AddPrefix(field="prediction", prefix="SELECT "),
            GetSQL(field="prediction"),
        ]
    ),
    "processors.text2sql.get_sql",
)
