"""This section describes unitxt operators for structured data.

These operators are specialized in handling structured data like tables.
For tables, expected input format is:
{
  "header": ["col1", "col2"],
  "rows": [["row11", "row12"], ["row21", "row22"], ["row31", "row32"]]
}

For triples, expected input format is:
[[ "subject1", "relation1", "object1" ], [ "subject1", "relation2", "object2"]]

For key-value pairs, expected input format is:
{"key1": "value1", "key2": value2, "key3": "value3"}
------------------------
"""

import json
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import pandas as pd

from .dict_utils import dict_get
from .operators import FieldOperator, InstanceOperator


class SerializeTable(ABC, FieldOperator):
    """TableSerializer converts a given table into a flat sequence with special symbols.

    Output format varies depending on the chosen serializer. This abstract class defines structure of a typical table serializer that any concrete implementation should follow.
    """

    # main method to serialize a table
    @abstractmethod
    def serialize_table(self, table_content: Dict) -> str:
        pass

    # method to process table header
    def process_header(self, header: List):
        pass

    # method to process a table row
    def process_row(self, row: List, row_index: int):
        pass


# Concrete classes implementing table serializers
class SerializeTableAsIndexedRowMajor(SerializeTable):
    """Indexed Row Major Table Serializer.

    Commonly used row major serialization format.
    Format:  col : col1 | col2 | col 3 row 1 : val1 | val2 | val3 | val4 row 2 : val1 | ...
    """

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


class SerializeTableAsMarkdown(SerializeTable):
    """Markdown Table Serializer.

    Markdown table format is used in GitHub code primarily.
    Format:
    |col1|col2|col3|
    |---|---|---|
    |A|4|1|
    |I|2|1|
    ...
    """

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


class SerializeTableAsDFLoader(SerializeTable):
    """DFLoader Table Serializer.

    Pandas dataframe based code snippet format serializer.
    Format(Sample):
    pd.DataFrame({
        "name" : ["Alex", "Diana", "Donald"],
        "age" : [26, 34, 39]
    },
    index=[0,1,2])
    """

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

        # Create a pandas DataFrame
        df = pd.DataFrame(rows, columns=header)

        # Generate output string in the desired format
        data_dict = df.to_dict(orient="list")

        return (
            "pd.DataFrame({\n"
            + json.dumps(data_dict)
            + "},\nindex="
            + str(list(range(len(rows))))
            + ")"
        )


class SerializeTableAsJson(SerializeTable):
    """JSON Table Serializer.

    Json format based serializer.
    Format(Sample):
    {
        "0":{"name":"Alex","age":26},
        "1":{"name":"Diana","age":34},
        "2":{"name":"Donald","age":39}
    }
    """

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

        # Generate output dictionary
        output_dict = {}
        for i, row in enumerate(rows):
            output_dict[i] = {header[j]: value for j, value in enumerate(row)}

        # Convert dictionary to JSON string
        return json.dumps(output_dict)


# truncate cell value to maximum allowed length
def truncate_cell(cell_value, max_len):
    if cell_value is None:
        return None

    if isinstance(cell_value, int) or isinstance(cell_value, float):
        return None

    if cell_value.strip() == "":
        return None

    if len(cell_value) > max_len:
        return cell_value[:max_len]

    return None


class TruncateTableCells(InstanceOperator):
    """Limit the maximum length of cell values in a table to reduce the overall length.

    Args:
        max_length (int) - maximum allowed length of cell values
        For tasks that produce a cell value as answer, truncating a cell value should be replicated
        with truncating the corresponding answer as well. This has been addressed in the implementation.

    """

    max_length: int = 15
    table: str = None
    text_output: Optional[str] = None

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        table = dict_get(instance, self.table)

        answers = []
        if self.text_output is not None:
            answers = dict_get(instance, self.text_output)

        self.truncate_table(table_content=table, answers=answers)

        return instance

    # truncate table cells
    def truncate_table(self, table_content: Dict, answers: Optional[List]):
        cell_mapping = {}

        # One row at a time
        for row in table_content.get("rows", []):
            for i, cell in enumerate(row):
                truncated_cell = truncate_cell(cell, self.max_length)
                if truncated_cell is not None:
                    cell_mapping[cell] = truncated_cell
                    row[i] = truncated_cell

        # Update values in answer list to truncated values
        if answers is not None:
            for i, case in enumerate(answers):
                answers[i] = cell_mapping.get(case, case)


class TruncateTableRows(FieldOperator):
    """Limits table rows to specified limit by removing excess rows via random selection.

    Args:
        rows_to_keep (int) - number of rows to keep.
    """

    rows_to_keep: int = 10

    def process_value(self, table: Any) -> Any:
        return self.truncate_table_rows(table_content=table)

    def truncate_table_rows(self, table_content: Dict):
        # Get rows from table
        rows = table_content.get("rows", [])

        num_rows = len(rows)

        # if number of rows are anyway lesser, return.
        if num_rows <= self.rows_to_keep:
            return table_content

        # calculate number of rows to delete, delete them
        rows_to_delete = num_rows - self.rows_to_keep

        # Randomly select rows to be deleted
        deleted_rows_indices = random.sample(range(len(rows)), rows_to_delete)

        remaining_rows = [
            row for i, row in enumerate(rows) if i not in deleted_rows_indices
        ]
        table_content["rows"] = remaining_rows

        return table_content


class SerializeTableRowAsText(InstanceOperator):
    """Serializes a table row as text.

    Args:
        fields (str) - list of fields to be included in serialization.
        to_field (str) - serialized text field name.
        max_cell_length (int) - limits cell length to be considered, optional.
    """

    fields: str
    to_field: str
    max_cell_length: Optional[int] = None

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        linearized_str = ""
        for field in self.fields:
            value = dict_get(instance, field)
            if self.max_cell_length is not None:
                truncated_value = truncate_cell(value, self.max_cell_length)
                if truncated_value is not None:
                    value = truncated_value

            linearized_str = linearized_str + field + " is " + str(value) + ", "

        instance[self.to_field] = linearized_str
        return instance


class SerializeTableRowAsList(InstanceOperator):
    """Serializes a table row as list.

    Args:
        fields (str) - list of fields to be included in serialization.
        to_field (str) - serialized text field name.
        max_cell_length (int) - limits cell length to be considered, optional.
    """

    fields: str
    to_field: str
    max_cell_length: Optional[int] = None

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        linearized_str = ""
        for field in self.fields:
            value = dict_get(instance, field)
            if self.max_cell_length is not None:
                truncated_value = truncate_cell(value, self.max_cell_length)
                if truncated_value is not None:
                    value = truncated_value

            linearized_str = linearized_str + field + ": " + str(value) + ", "

        instance[self.to_field] = linearized_str
        return instance


class SerializeTriples(FieldOperator):
    """Serializes triples into a flat sequence.

    Sample input in expected format:
    [[ "First Clearing", "LOCATION", "On NYS 52 1 Mi. Youngsville" ], [ "On NYS 52 1 Mi. Youngsville", "CITY_OR_TOWN", "Callicoon, New York"]]

    Sample output:
    First Clearing : LOCATION : On NYS 52 1 Mi. Youngsville | On NYS 52 1 Mi. Youngsville : CITY_OR_TOWN : Callicoon, New York

    """

    def process_value(self, tripleset: Any) -> Any:
        return self.serialize_triples(tripleset)

    def serialize_triples(self, tripleset) -> str:
        return " | ".join(
            f"{subj} : {rel.lower()} : {obj}" for subj, rel, obj in tripleset
        )


class SerializeKeyValPairs(FieldOperator):
    """Serializes key, value pairs into a flat sequence.

    Sample input in expected format: {"name": "Alex", "age": 31, "sex": "M"}
    Sample output: name is Alex, age is 31, sex is M
    """

    def process_value(self, kvpairs: Any) -> Any:
        return self.serialize_kvpairs(kvpairs)

    def serialize_kvpairs(self, kvpairs) -> str:
        serialized_str = ""
        for key, value in kvpairs.items():
            serialized_str += f"{key} is {value}, "

        # Remove the trailing comma and space then return
        return serialized_str[:-2]


class ListToKeyValPairs(InstanceOperator):
    """Maps list of keys and values into key:value pairs.

    Sample input in expected format: {"keys": ["name", "age", "sex"], "values": ["Alex", 31, "M"]}
    Sample output: {"name": "Alex", "age": 31, "sex": "M"}
    """

    fields: List[str]
    to_field: str

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        keylist = dict_get(instance, self.fields[0])
        valuelist = dict_get(instance, self.fields[1])

        output_dict = {}
        for key, value in zip(keylist, valuelist):
            output_dict[key] = value

        instance[self.to_field] = output_dict

        return instance


class ConvertTableColNamesToSequential(FieldOperator):
    """Replaces actual table column names with static sequential names like col_0, col_1,...

    Sample input:
    {
        "header": ["name", "age"],
        "rows": [["Alex", 21], ["Donald", 34]]
    }
    Sample output:
    {
        "header": ["col_0", "col_1"],
        "rows": [["Alex", 21], ["Donald", 34]]
    }
    """

    def process_value(self, table: Any) -> Any:
        table_input = deepcopy(table)
        return self.replace_header(table_content=table_input)

    # replaces header with sequential column names
    def replace_header(self, table_content: Dict) -> str:
        # Extract header from the dictionary
        header = table_content.get("header", [])

        assert header, "Input table missing header"

        new_header = ["col_" + str(i) for i in range(len(header))]
        table_content["header"] = new_header

        return table_content


class ShuffleTableRows(FieldOperator):
    """Shuffles the input table rows randomly.

    Sample Input:
    {
        "header": ["name", "age"],
        "rows": [["Alex", 26], ["Raj", 34], ["Donald", 39]],
    }

    Sample Output:
    {
        "header": ["name", "age"],
        "rows": [["Donald", 39], ["Raj", 34], ["Alex", 26]],
    }
    """

    def process_value(self, table: Any) -> Any:
        table_input = deepcopy(table)
        return self.shuffle_rows(table_content=table_input)

    # shuffles table rows randomly
    def shuffle_rows(self, table_content: Dict) -> str:
        # extract header & rows from the dictionary
        header = table_content.get("header", [])
        rows = table_content.get("rows", [])
        assert header and rows, "Incorrect input table format"

        # shuffle rows
        random.shuffle(rows)
        table_content["rows"] = rows

        return table_content


class ShuffleTableColumns(FieldOperator):
    """Shuffles the table columns randomly.

    Sample Input:
        {
            "header": ["name", "age"],
            "rows": [["Alex", 26], ["Raj", 34], ["Donald", 39]],
        }

    Sample Output:
        {
            "header": ["age", "name"],
            "rows": [[26, "Alex"], [34, "Raj"], [39, "Donald"]],
        }
    """

    def process_value(self, table: Any) -> Any:
        table_input = deepcopy(table)
        return self.shuffle_columns(table_content=table_input)

    # shuffles table columns randomly
    def shuffle_columns(self, table_content: Dict) -> str:
        # extract header & rows from the dictionary
        header = table_content.get("header", [])
        rows = table_content.get("rows", [])
        assert header and rows, "Incorrect input table format"

        # shuffle the indices first
        indices = list(range(len(header)))
        random.shuffle(indices)  #

        # shuffle the header & rows based on that indices
        shuffled_header = [header[i] for i in indices]
        shuffled_rows = [[row[i] for i in indices] for row in rows]

        table_content["header"] = shuffled_header
        table_content["rows"] = shuffled_rows

        return table_content


class LoadJson(FieldOperator):
    failure_value: Any = None
    allow_failure: bool = False

    def process_value(self, value: str) -> Any:
        if self.allow_failure:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return self.failure_value
        else:
            return json.loads(value)


class DumpJson(FieldOperator):
    def process_value(self, value: str) -> str:
        return json.dumps(value)


class MapHTMLTableToJSON(FieldOperator):
    """Converts HTML table format to the basic one (JSON).

    JSON format
    {
        "header": ["col1", "col2"],
        "rows": [["row11", "row12"], ["row21", "row22"], ["row31", "row32"]]
    }
    """

    _requirements_list = ["bs4"]

    def process_value(self, table: Any) -> Any:
        return self.truncate_table_rows(table_content=table)

    def truncate_table_rows(self, table_content: str) -> Dict:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(table_content, "html.parser")

        # Extract header
        header = []
        header_cells = soup.find("thead").find_all("th")
        for cell in header_cells:
            header.append(cell.get_text())

        # Extract rows
        rows = []
        for row in soup.find("tbody").find_all("tr"):
            row_data = []
            for cell in row.find_all("td"):
                row_data.append(cell.get_text())
            rows.append(row_data)

        # return dictionary

        return {"header": header, "rows": rows}
