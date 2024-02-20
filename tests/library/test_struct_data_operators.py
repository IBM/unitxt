import unittest

from unitxt.struct_data_operators import (
    ListToKeyValPairs,
    SerializeKeyValPairs,
    SerializeTableAsIndexedRowMajor,
    SerializeTableAsMarkdown,
    SerializeTableRowAsList,
    SerializeTableRowAsText,
    SerializeTriples,
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

    def test_serializetablerow_as_text(self):
        inputs = [
            {
                "name": "Alex",
                "age": "26",
            }
        ]

        serialized_str = "name is Alex, age is 26, "

        targets = [
            {
                "name": "Alex",
                "age": "26",
                "serialized_row": serialized_str,
            },
        ]

        check_operator(
            operator=SerializeTableRowAsText(
                fields=["name", "age"], to_field="serialized_row", max_cell_length=25
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_serializetablerow_as_list(self):
        inputs = [
            {
                "name": "Alex",
                "age": "26",
            }
        ]

        serialized_str = "name: Alex, age: 26, "

        targets = [
            {
                "name": "Alex",
                "age": "26",
                "serialized_row": serialized_str,
            },
        ]

        check_operator(
            operator=SerializeTableRowAsList(
                fields=["name", "age"], to_field="serialized_row", max_cell_length=25
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_serialize_triples(self):
        inputs = [
            {
                "tripleset": [
                    ["First Clearing", "LOCATION", "On NYS 52 1 Mi. Youngsville"],
                    [
                        "On NYS 52 1 Mi. Youngsville",
                        "CITY_OR_TOWN",
                        "Callicoon, New York",
                    ],
                ]
            }
        ]

        serialized_str = "First Clearing : location : On NYS 52 1 Mi. Youngsville | On NYS 52 1 Mi. Youngsville : city_or_town : Callicoon, New York"

        targets = [
            {
                "tripleset": [
                    ["First Clearing", "LOCATION", "On NYS 52 1 Mi. Youngsville"],
                    [
                        "On NYS 52 1 Mi. Youngsville",
                        "CITY_OR_TOWN",
                        "Callicoon, New York",
                    ],
                ],
                "serialized_triples": serialized_str,
            },
        ]

        check_operator(
            operator=SerializeTriples(
                field_to_field={"tripleset": "serialized_triples"}
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_serialize_kvpairs(self):
        inputs = [{"kvpairs": {"name": "Alex", "age": 31, "sex": "M"}}]

        serialized_str = "name is Alex, age is 31, sex is M"

        targets = [
            {
                "kvpairs": {"name": "Alex", "age": 31, "sex": "M"},
                "serialized_kvpairs": serialized_str,
            },
        ]

        check_operator(
            operator=SerializeKeyValPairs(
                field_to_field={"kvpairs": "serialized_kvpairs"}
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_list_to_kvpairs(self):
        inputs = [{"keys": ["name", "age", "sex"], "values": ["Alex", 31, "M"]}]

        targets = [
            {
                "keys": ["name", "age", "sex"],
                "values": ["Alex", 31, "M"],
                "kvpairs": {"name": "Alex", "age": 31, "sex": "M"},
            },
        ]

        check_operator(
            operator=ListToKeyValPairs(
                fields=["keys", "values"], to_field="kvpairs", use_query=True
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
