from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (
    Any,
    Dict,
    List,
)

from .operators import FieldOperator

"""
TableSerializer converts a given table into a flat sequence with special symbols.
Input table format must be:
{"header": ["col1", "col2"], "rows": [["row11", "row12"], ["row21", "row22"], ["row31", "row32"]]}
Output format varies depending on the chosen serializer. Abstract class at the top defines structure of a typical table serializer that any concrete implementation should follow.
"""


class TableSerializer(ABC, FieldOperator):
    # main method to serialize a table
    @abstractmethod
    def serialize_table(self, table_content: Dict) -> str:
        pass

    # method to process table header
    @abstractmethod
    def process_header(self, header: List):
        pass

    # method to process a table row
    @abstractmethod
    def process_row(self, row: List, row_index: int):
        pass


# Concrete classes implementing table serializers follow..
"""
Indexed Row Major Table Serializer.
Commonly used row major serialization format.
Format:  col : col1 | col2 | col 3 row 1 : val1 | val2 | val3 | val4 row 2 : val1 | ...
"""


class IndexedRowMajorTableSerializer(TableSerializer):
    def process_value(self, table: Any) -> Any:
        table_input = deepcopy(table)
        return self.serialize_table(table_content=table_input)

    # main method that processes a table
    # table_content must be in the presribed input format
    def serialize_table(self, table_content: Dict) -> str:
        # Extract headers and rows from the dictionary
        header = table_content.get("header", [])
        rows = table_content.get("rows", [])

        assert header and rows, "Incorrect input table format"

        # Process table header first
        serialized_tbl_str = self.process_header(header) + " "

        # Process rows sequentially starting from row 1
        for i, row in enumerate(rows, start=1):
            serialized_tbl_str += self.process_row(row, row_index=i) + " "

        # return serialized table as a string
        return serialized_tbl_str.strip()

    # serialize header into a string containing the list of column names separated by '|' symbol
    def process_header(self, header: List):
        return "col : " + " | ".join(header)

    # serialize a table row into a string containing the list of cell values separated by '|'
    def process_row(self, row: List, row_index: int):
        serialized_row_str = ""
        row_cell_values = [
            str(value) if isinstance(value, (int, float)) else value for value in row
        ]

        serialized_row_str += " | ".join(row_cell_values)

        return f"row {row_index} : {serialized_row_str}"


"""
Markdown Table Serializer.
Markdown table format is used in GitHub code primarily.
Format:
|col1|col2|col3|
|---|---|---|
|A|4|1|
|I|2|1|
...
"""


class MarkdownTableSerializer(TableSerializer):
    def process_value(self, table: Any) -> Any:
        table_input = deepcopy(table)
        return self.serialize_table(table_content=table_input)

    # main method that serializes a table.
    # table_content must be in the presribed input format.
    def serialize_table(self, table_content: Dict) -> str:
        # Extract headers and rows from the dictionary
        header = table_content.get("header", [])
        rows = table_content.get("rows", [])

        assert header and rows, "Incorrect input table format"

        # Process table header first
        serialized_tbl_str = self.process_header(header)

        # Process rows sequentially starting from row 1
        for i, row in enumerate(rows, start=1):
            serialized_tbl_str += self.process_row(row, row_index=i)

        # return serialized table as a string
        return serialized_tbl_str.strip()

    # serialize header into a string containing the list of column names
    def process_header(self, header: List):
        header_str = "|{}|\n".format("|".join(header))
        header_str += "|{}|\n".format("|".join(["---"] * len(header)))
        return header_str

    # serialize a table row into a string containing the list of cell values
    def process_row(self, row: List, row_index: int):
        row_str = ""
        row_str += "|{}|\n".format("|".join(str(cell) for cell in row))
        return row_str
