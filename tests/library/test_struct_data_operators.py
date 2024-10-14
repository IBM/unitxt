from unitxt.struct_data_operators import (
    ConvertTableColNamesToSequential,
    DumpJson,
    DuplicateTableColumns,
    DuplicateTableRows,
    InsertEmptyTableRows,
    ListToKeyValPairs,
    LoadJson,
    MapHTMLTableToJSON,
    SerializeKeyValPairs,
    SerializeTableAsDFLoader,
    SerializeTableAsHTML,
    SerializeTableAsIndexedRowMajor,
    SerializeTableAsJson,
    SerializeTableAsMarkdown,
    SerializeTableRowAsList,
    SerializeTableRowAsText,
    SerializeTriples,
    ShuffleTableColumns,
    ShuffleTableRows,
    TransposeTable,
    TruncateTableCells,
    TruncateTableRows,
)
from unitxt.test_utils.operators import (
    check_operator,
)

from tests.utils import UnitxtTestCase


class TestStructDataOperators(UnitxtTestCase):
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

    def test_serializetable_markdown_with_shuffle(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", "26"], ["Raj", "34"], ["Donald", "39"]],
                }
            }
        ]

        serialized_str = "|age|name|\n|---|---|\n|39|Donald|\n|34|Raj|\n|26|Alex|"

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
                field_to_field={"table": "serialized_table"},
                shuffle_columns=True,
                shuffle_rows=True,
                seed=1,
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

    def test_serializetable_dfloader(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", 26], ["Raj", 34], ["Donald", 39]],
                }
            }
        ]

        serialized_str = 'pd.DataFrame({\n{"name": ["Alex", "Raj", "Donald"], "age": [26, 34, 39]}},\nindex=[0, 1, 2])'

        targets = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", 26], ["Raj", 34], ["Donald", 39]],
                },
                "serialized_table": serialized_str,
            }
        ]

        check_operator(
            operator=SerializeTableAsDFLoader(
                field_to_field={"table": "serialized_table"}
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_serializetable_json(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", 26], ["Raj", 34], ["Donald", 39]],
                }
            }
        ]

        serialized_str = '{"0": {"name": "Alex", "age": 26}, "1": {"name": "Raj", "age": 34}, "2": {"name": "Donald", "age": 39}}'

        targets = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", 26], ["Raj", 34], ["Donald", 39]],
                },
                "serialized_table": serialized_str,
            }
        ]

        check_operator(
            operator=SerializeTableAsJson(field_to_field={"table": "serialized_table"}),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_serializetable_html(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", 26], ["Raj", 34], ["Donald", 39]],
                }
            }
        ]

        serialized_str = "<table>\n  <thead>\n    <tr><th>name</th><th>age</th></tr>\n  </thead>\n  <tbody>\n    <tr><td>Alex</td><td>26</td></tr>\n    <tr><td>Raj</td><td>34</td></tr>\n    <tr><td>Donald</td><td>39</td></tr>\n  </tbody>\n</table>"

        targets = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", 26], ["Raj", 34], ["Donald", 39]],
                },
                "serialized_table": serialized_str,
            }
        ]

        check_operator(
            operator=SerializeTableAsHTML(field_to_field={"table": "serialized_table"}),
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
            operator=ListToKeyValPairs(fields=["keys", "values"], to_field="kvpairs"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_convert_table_colnames_to_sequential_names(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [
                        ["Alex", 21],
                        ["Donald", 34],
                    ],
                }
            }
        ]

        targets = [
            {
                "table": {
                    "header": ["col_0", "col_1"],
                    "rows": [
                        ["Alex", 21],
                        ["Donald", 34],
                    ],
                }
            }
        ]

        check_operator(
            operator=ConvertTableColNamesToSequential(field="table"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_shuffle_table_rows(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [
                        ["Alex", 21],
                        ["Raj", 34],
                        ["Donald", 39],
                    ],
                }
            }
        ]

        targets = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Raj", 34], ["Alex", 21], ["Donald", 39]],
                }
            }
        ]

        check_operator(
            operator=ShuffleTableRows(field="table"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_shuffle_table_columns(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [
                        ["Alex", 21],
                        ["Raj", 34],
                        ["Donald", 39],
                    ],
                }
            }
        ]

        targets = [
            {
                "table": {
                    "header": ["age", "name"],
                    "rows": [
                        [21, "Alex"],
                        [34, "Raj"],
                        [39, "Donald"],
                    ],
                }
            }
        ]

        import random

        random.seed(123)

        check_operator(
            operator=ShuffleTableColumns(field="table"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_load_and_dump_json(self):
        inputs = [{"data": {"hello": "world"}}]

        targets = [{"data": '{"hello": "world"}'}]

        check_operator(
            operator=DumpJson(field="data"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        check_operator(
            operator=LoadJson(field="data"),
            inputs=targets,
            targets=inputs,
            tester=self,
        )

    def test_load_json_failures(self):
        inputs = [{"data": '{"hello": world"}'}]

        targets = [{"data": None}]

        check_operator(
            operator=LoadJson(field="data", allow_failure=True),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        with self.assertRaises(ValueError):
            check_operator(
                operator=LoadJson(field="data", allow_failure=False),
                inputs=inputs,
                targets=targets,
                tester=self,
            )

    def test_map_htmltable_to_json(self):
        inputs = [
            {
                "data": "<table border='1' class='dataframe'> <thead> <tr style='text-align: right;'> <th></th> <th>F1</th> </tr> </thead> <tbody> <tr> <td>Position Feature || plain text PF</td> <td>83.21</td> </tr> <tr> <td>Position Feature || TPF1</td> <td>83.99</td> </tr> </tbody></table>"
            }
        ]
        targets = [
            {
                "data": "<table border='1' class='dataframe'> <thead> <tr style='text-align: right;'> <th></th> <th>F1</th> </tr> </thead> <tbody> <tr> <td>Position Feature || plain text PF</td> <td>83.21</td> </tr> <tr> <td>Position Feature || TPF1</td> <td>83.99</td> </tr> </tbody></table>",
                "table_out": {
                    "header": ["", "F1"],
                    "rows": [
                        ["Position Feature || plain text PF", "83.21"],
                        ["Position Feature || TPF1", "83.99"],
                    ],
                },
            }
        ]

        check_operator(
            operator=MapHTMLTableToJSON(field_to_field=[["data", "table_out"]]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_transpose_table(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age", "sex"],
                    "rows": [["Alice", 26, "F"], ["Raj", 34, "M"], ["Donald", 39, "M"]],
                }
            }
        ]

        targets = [
            {
                "table": {
                    "header": [" ", "0", "1", "2"],
                    "rows": [
                        ["name", "Alice", "Raj", "Donald"],
                        ["age", 26, 34, 39],
                        ["sex", "F", "M", "M"],
                    ],
                }
            }
        ]

        check_operator(
            operator=TransposeTable(field="table"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_duplicate_table_rows(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age", "sex"],
                    "rows": [["Alice", 26, "F"], ["Raj", 34, "M"], ["Donald", 39, "M"]],
                }
            }
        ]

        targets = [
            {
                "table": {
                    "header": ["name", "age", "sex"],
                    "rows": [
                        ["Alice", 26, "F"],
                        ["Alice", 26, "F"],  # First row duplicated
                        ["Raj", 34, "M"],  # Second row remains the same
                        ["Donald", 39, "M"],
                        ["Donald", 39, "M"],  # Third row duplicated
                    ],
                }
            }
        ]

        check_operator(
            operator=DuplicateTableRows(field="table", row_indices=[0, 2], times=2),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_duplicate_table_columns(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age", "sex"],
                    "rows": [["Alice", 26, "F"], ["Raj", 34, "M"], ["Donald", 39, "M"]],
                }
            }
        ]

        targets = [
            {
                "table": {
                    "header": ["name", "name", "age", "age", "sex"],
                    "rows": [
                        ["Alice", "Alice", 26, 26, "F"],
                        ["Raj", "Raj", 34, 34, "M"],
                        ["Donald", "Donald", 39, 39, "M"],
                    ],
                }
            }
        ]

        check_operator(
            operator=DuplicateTableColumns(
                field="table", column_indices=[0, 1], times=2
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_insert_empty_table_rows(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age", "sex"],
                    "rows": [["Alice", 26, "F"], ["Raj", 34, "M"], ["Donald", 39, "M"]],
                }
            }
        ]

        targets = [
            {
                "table": {
                    "header": ["name", "age", "sex"],
                    "rows": [
                        ["", "", ""],  # Empty row inserted
                        ["Alice", 26, "F"],
                        ["", "", ""],  # Empty row inserted
                        ["Raj", 34, "M"],
                        ["Donald", 39, "M"],
                    ],
                }
            }
        ]

        import random

        random.seed(123)

        check_operator(
            operator=InsertEmptyTableRows(field="table", times=2),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
