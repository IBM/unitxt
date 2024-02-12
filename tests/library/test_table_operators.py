import unittest

from src.unitxt.table_operators import (
    SerializeTableAsIndexedRowMajor,
    SerializeTableAsMarkdown,
    TruncateTableCells,
    TruncateTableRows,
)
from src.unitxt.test_utils.operators import (
    check_operator,
)


class TestTableOperators(unittest.TestCase):
    """Tests for tabular data processing operators."""

    def test_serializetable_markdown(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", "26"], ["Raj", "34"], ["Donald", "39"]],
                }
            }
        ]

        serialized_str = "|name|age|\n|---|---|\n|Alex|26|\n|Raj|34|\n|Donald|39|"

        targets = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", "26"], ["Raj", "34"], ["Donald", "39"]],
                },
                "serialized_table": serialized_str,
            }
        ]

        check_operator(
            operator=SerializeTableAsMarkdown(
                field_to_field={"table": "serialized_table"}
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_serializetable_indexedrowmajor(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", "26"], ["Raj", "34"], ["Donald", "39"]],
                }
            }
        ]

        serialized_str = (
            "col : name | age row 1 : Alex | 26 row 2 : Raj | 34 row 3 : Donald | 39"
        )

        targets = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", "26"], ["Raj", "34"], ["Donald", "39"]],
                },
                "serialized_table": serialized_str,
            }
        ]

        check_operator(
            operator=SerializeTableAsIndexedRowMajor(
                field_to_field={"table": "serialized_table"}
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_truncate_table_cells(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age", "comment"],
                    "rows": [
                        ["Alex", "26", "works in XYZ project, has 10 years exp."],
                        [
                            "Raj",
                            "34",
                            "works in Project M, primary work is around FMs.",
                        ],
                        ["Donald", "39", "works in KBM project, has 3 years exp."],
                    ],
                }
            }
        ]

        targets = [
            {
                "table": {
                    "header": ["name", "age", "comment"],
                    "rows": [
                        ["Alex", "26", "works in XYZ project"],
                        ["Raj", "34", "works in Project M, "],
                        ["Donald", "39", "works in KBM project"],
                    ],
                }
            }
        ]

        check_operator(
            operator=TruncateTableCells(max_length=20, table="table"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_truncate_table_rows(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", "26"]],
                }
            }
        ]

        targets = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [],
                }
            }
        ]

        check_operator(
            operator=TruncateTableRows(field="table", rows_to_keep=0),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
