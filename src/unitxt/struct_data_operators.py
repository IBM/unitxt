"""This section describes unitxt operators for structured data.

These operators are specialized in handling structured data like tables.
For tables, expected input format is:

.. code-block:: text

    {
        "header": ["col1", "col2"],
        "rows": [["row11", "row12"], ["row21", "row22"], ["row31", "row32"]]
    }

For triples, expected input format is:

.. code-block:: text

    [[ "subject1", "relation1", "object1" ], [ "subject1", "relation2", "object2"]]

For key-value pairs, expected input format is:

.. code-block:: text

    {"key1": "value1", "key2": value2, "key3": "value3"}
"""

import json
import random
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import pandas as pd

from .augmentors import TypeDependentAugmentor
from .dict_utils import dict_get
from .operators import FieldOperator, InstanceOperator
from .random_utils import new_random_generator
from .serializers import ImageSerializer, TableSerializer
from .types import Table
from .utils import recursive_copy


def shuffle_columns(table: Table, seed=0) -> Table:
    # extract header & rows from the dictionary
    header = table.get("header", [])
    rows = table.get("rows", [])
    # shuffle the indices first
    indices = list(range(len(header)))
    random_generator = new_random_generator({"table": table, "seed": seed})
    random_generator.shuffle(indices)

    # shuffle the header & rows based on that indices
    shuffled_header = [header[i] for i in indices]
    shuffled_rows = [[row[i] for i in indices] for row in rows]

    table["header"] = shuffled_header
    table["rows"] = shuffled_rows

    return table


def shuffle_rows(table: Table, seed=0) -> Table:
    # extract header & rows from the dictionary
    rows = table.get("rows", [])
    # shuffle rows
    random_generator = new_random_generator({"table": table, "seed": seed})
    random_generator.shuffle(rows)
    table["rows"] = rows

    return table


class SerializeTable(ABC, TableSerializer):
    """TableSerializer converts a given table into a flat sequence with special symbols.

    Output format varies depending on the chosen serializer. This abstract class defines structure of a typical table serializer that any concrete implementation should follow.
    """

    seed: int = 0
    shuffle_rows: bool = False
    shuffle_columns: bool = False

    def serialize(self, value: Table, instance: Dict[str, Any]) -> str:
        value = recursive_copy(value)
        if self.shuffle_columns:
            value = shuffle_columns(table=value, seed=self.seed)

        if self.shuffle_rows:
            value = shuffle_rows(table=value, seed=self.seed)

        return self.serialize_table(value)

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

    .. code-block:: text

        |col1|col2|col3|
        |---|---|---|
        |A|4|1|
        |I|2|1|
        ...

    """

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

    .. code-block:: python

        pd.DataFrame({
            "name" : ["Alex", "Diana", "Donald"],
            "age" : [26, 34, 39]
        },
        index=[0,1,2])
    """

    # main method that serializes a table.
    # table_content must be in the presribed input format.
    def serialize_table(self, table_content: Dict) -> str:
        # Extract headers and rows from the dictionary
        header = table_content.get("header", [])
        rows = table_content.get("rows", [])

        assert header and rows, "Incorrect input table format"

        # Fix duplicate columns, ensuring the first occurrence has no suffix
        header = [
            f"{col}_{header[:i].count(col)}" if header[:i].count(col) > 0 else col
            for i, col in enumerate(header)
        ]

        # Create a pandas DataFrame
        df = pd.DataFrame(rows, columns=header)

        # Generate output string in the desired format
        data_dict = df.to_dict(orient="list")

        return (
            "pd.DataFrame({\n"
            + json.dumps(data_dict)[1:-1]
            + "},\nindex="
            + str(list(range(len(rows))))
            + ")"
        )


class SerializeTableAsJson(SerializeTable):
    """JSON Table Serializer.

    Json format based serializer.
    Format(Sample):

    .. code-block:: json

        {
            "0":{"name":"Alex","age":26},
            "1":{"name":"Diana","age":34},
            "2":{"name":"Donald","age":39}
        }
    """

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


class SerializeTableAsHTML(SerializeTable):
    """HTML Table Serializer.

    HTML table format used for rendering tables in web pages.
    Format(Sample):

    .. code-block:: html

        <table>
            <thead>
                <tr><th>name</th><th>age</th><th>sex</th></tr>
            </thead>
            <tbody>
                <tr><td>Alice</td><td>26</td><td>F</td></tr>
                <tr><td>Raj</td><td>34</td><td>M</td></tr>
            </tbody>
        </table>
    """

    # main method that serializes a table.
    # table_content must be in the prescribed input format.
    def serialize_table(self, table_content: Dict) -> str:
        # Extract headers and rows from the dictionary
        header = table_content.get("header", [])
        rows = table_content.get("rows", [])

        assert header and rows, "Incorrect input table format"

        # Build the HTML table structure
        serialized_tbl_str = "<table>\n"
        serialized_tbl_str += self.process_header(header) + "\n"
        serialized_tbl_str += self.process_rows(rows) + "\n"
        serialized_tbl_str += "</table>"

        return serialized_tbl_str.strip()

    # serialize the header into an HTML <thead> section
    def process_header(self, header: List) -> str:
        header_html = "  <thead>\n    <tr>"
        for col in header:
            header_html += f"<th>{col}</th>"
        header_html += "</tr>\n  </thead>"
        return header_html

    # serialize the rows into an HTML <tbody> section
    def process_rows(self, rows: List[List]) -> str:
        rows_html = "  <tbody>"
        for row in rows:
            rows_html += "\n    <tr>"
            for cell in row:
                rows_html += f"<td>{cell}</td>"
            rows_html += "</tr>"
        rows_html += "\n  </tbody>"
        return rows_html


class SerializeTableAsConcatenation(SerializeTable):
    """Concat Serializer.

    Concat all table content to one string of header and rows.
    Format(Sample):
    name age Alex 26 Diana 34
    """

    def serialize_table(self, table_content: Dict) -> str:
        # Extract headers and rows from the dictionary
        header = table_content["header"]
        rows = table_content["rows"]

        assert header and rows, "Incorrect input table format"

        # Process table header first
        serialized_tbl_str = " ".join([str(i) for i in header])

        # Process rows sequentially starting from row 1
        for row in rows:
            serialized_tbl_str += " " + " ".join([str(i) for i in row])

        # return serialized table as a string
        return serialized_tbl_str.strip()


class SerializeTableAsImage(SerializeTable):
    _requirements_list = ["matplotlib", "pillow"]

    def serialize_table(self, table_content: Dict) -> str:
        raise NotImplementedError()

    def serialize(self, value: Table, instance: Dict[str, Any]) -> str:
        table_content = recursive_copy(value)
        if self.shuffle_columns:
            table_content = shuffle_columns(table=table_content, seed=self.seed)

        if self.shuffle_rows:
            table_content = shuffle_rows(table=table_content, seed=self.seed)

        import io

        import matplotlib.pyplot as plt
        import pandas as pd
        from PIL import Image

        # Extract headers and rows from the dictionary
        header = table_content.get("header", [])
        rows = table_content.get("rows", [])

        assert header and rows, "Incorrect input table format"

        # Fix duplicate columns, ensuring the first occurrence has no suffix
        header = [
            f"{col}_{header[:i].count(col)}" if header[:i].count(col) > 0 else col
            for i, col in enumerate(header)
        ]

        # Create a pandas DataFrame
        df = pd.DataFrame(rows, columns=header)

        # Fix duplicate columns, ensuring the first occurrence has no suffix
        df.columns = [
            f"{col}_{i}" if df.columns.duplicated()[i] else col
            for i, col in enumerate(df.columns)
        ]

        # Create a matplotlib table
        plt.rcParams["font.family"] = "Serif"
        fig, ax = plt.subplots(figsize=(len(header) * 1.5, len(rows) * 0.5))
        ax.axis("off")  # Turn off the axes

        table = pd.plotting.table(ax, df, loc="center", cellLoc="center")
        table.auto_set_column_width(col=range(len(df.columns)))
        table.scale(1.5, 1.5)

        # Save the plot to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)  # Close the figure to free up memory
        buf.seek(0)

        # Load the image from the buffer using PIL
        image = Image.open(buf)
        return ImageSerializer().serialize({"image": image, "format": "png"}, instance)


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
        rows_to_keep (int): number of rows to keep.
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

    .. code-block:: text

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
        table_input = recursive_copy(table)
        return self.replace_header(table_content=table_input)

    # replaces header with sequential column names
    def replace_header(self, table_content: Dict) -> str:
        # Extract header from the dictionary
        header = table_content.get("header", [])

        assert header, "Input table missing header"

        new_header = ["col_" + str(i) for i in range(len(header))]
        table_content["header"] = new_header

        return table_content


class ShuffleTableRows(TypeDependentAugmentor):
    """Shuffles the input table rows randomly.

    .. code-block:: text

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

    augmented_type = Table
    seed = 0

    def process_value(self, table: Any) -> Any:
        table_input = recursive_copy(table)
        return shuffle_rows(table_input, self.seed)


class ShuffleTableColumns(TypeDependentAugmentor):
    """Shuffles the table columns randomly.

    .. code-block:: text

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

    augmented_type = Table
    seed = 0

    def process_value(self, table: Any) -> Any:
        table_input = recursive_copy(table)
        return shuffle_columns(table_input, self.seed)


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
            return json.loads(value, strict=False)


class DumpJson(FieldOperator):
    def process_value(self, value: str) -> str:
        return json.dumps(value)


class MapHTMLTableToJSON(FieldOperator):
    """Converts HTML table format to the basic one (JSON).

    JSON format:

    .. code-block:: json

        {
            "header": ["col1", "col2"],
            "rows": [["row11", "row12"], ["row21", "row22"], ["row31", "row32"]]
        }
    """

    _requirements_list = ["bs4"]

    def process_value(self, table: Any) -> Any:
        return self.convert_to_json(table_content=table)

    def convert_to_json(self, table_content: str) -> Dict:
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


class MapTableListsToStdTableJSON(FieldOperator):
    """Converts lists table format to the basic one (JSON).

    JSON format:

    .. code-block:: json

        {
            "header": ["col1", "col2"],
            "rows": [["row11", "row12"], ["row21", "row22"], ["row31", "row32"]]
        }
    """

    def process_value(self, table: Any) -> Any:
        return self.map_tablelists_to_stdtablejson_util(table_content=table)

    def map_tablelists_to_stdtablejson_util(self, table_content: str) -> Dict:
        return {"header": table_content[0], "rows": table_content[1:]}


class ConstructTableFromRowsCols(InstanceOperator):
    """Maps column and row field into single table field encompassing both header and rows.

    field[0] = header string as List
    field[1] = rows string as List[List]
    field[2] = table caption string(optional)
    """

    fields: List[str]
    to_field: str

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        header = dict_get(instance, self.fields[0])
        rows = dict_get(instance, self.fields[1])

        if len(self.fields) >= 3:
            caption = instance[self.fields[2]]
        else:
            caption = None

        import ast

        header_processed = ast.literal_eval(header)
        rows_processed = ast.literal_eval(rows)

        output_dict = {"header": header_processed, "rows": rows_processed}

        if caption is not None:
            output_dict["caption"] = caption

        instance[self.to_field] = output_dict

        return instance


class TransposeTable(TypeDependentAugmentor):
    """Transpose a table.

    .. code-block:: text

        Sample Input:
            {
                "header": ["name", "age", "sex"],
                "rows": [["Alice", 26, "F"], ["Raj", 34, "M"], ["Donald", 39, "M"]],
            }

        Sample Output:
            {
                "header": [" ", "0", "1", "2"],
                "rows": [["name", "Alice", "Raj", "Donald"], ["age", 26, 34, 39], ["sex", "F", "M", "M"]],
            }

    """

    augmented_type = Table

    def process_value(self, table: Any) -> Any:
        return self.transpose_table(table)

    def transpose_table(self, table: Dict) -> Dict:
        # Extract the header and rows from the table object
        header = table["header"]
        rows = table["rows"]

        # Transpose the table by converting rows as columns and vice versa
        transposed_header = [" "] + [str(i) for i in range(len(rows))]
        transposed_rows = [
            [header[i]] + [row[i] for row in rows] for i in range(len(header))
        ]

        return {"header": transposed_header, "rows": transposed_rows}


class DuplicateTableRows(TypeDependentAugmentor):
    """Duplicates specific rows of a table for the given number of times.

    Args:
        row_indices (List[int]): rows to be duplicated

        times(int): each row to be duplicated is to show that many times
    """

    augmented_type = Table

    row_indices: List[int] = []
    times: int = 1

    def process_value(self, table: Any) -> Any:
        # Extract the header and rows from the table
        header = table["header"]
        rows = table["rows"]

        # Duplicate only the specified rows
        duplicated_rows = []
        for i, row in enumerate(rows):
            if i in self.row_indices:
                duplicated_rows.extend(
                    [row] * self.times
                )  # Duplicate the selected rows
            else:
                duplicated_rows.append(row)  # Leave other rows unchanged

        # Return the new table with selectively duplicated rows
        return {"header": header, "rows": duplicated_rows}


class DuplicateTableColumns(TypeDependentAugmentor):
    """Duplicates specific columns of a table for the given number of times.

    Args:
        column_indices (List[int]): columns to be duplicated

        times(int): each column to be duplicated is to show that many times
    """

    augmented_type = Table

    column_indices: List[int] = []
    times: int = 1

    def process_value(self, table: Any) -> Any:
        # Extract the header and rows from the table
        header = table["header"]
        rows = table["rows"]

        # Duplicate the specified columns in the header
        duplicated_header = []
        for i, col in enumerate(header):
            if i in self.column_indices:
                duplicated_header.extend([col] * self.times)
            else:
                duplicated_header.append(col)

        # Duplicate the specified columns in each row
        duplicated_rows = []
        for row in rows:
            new_row = []
            for i, value in enumerate(row):
                if i in self.column_indices:
                    new_row.extend([value] * self.times)
                else:
                    new_row.append(value)
            duplicated_rows.append(new_row)

        # Return the new table with selectively duplicated columns
        return {"header": duplicated_header, "rows": duplicated_rows}


class InsertEmptyTableRows(TypeDependentAugmentor):
    """Inserts empty rows in a table randomly for the given number of times.

    Args:
        times(int) - how many times to insert
    """

    augmented_type = Table

    times: int = 0

    def process_value(self, table: Any) -> Any:
        # Extract the header and rows from the table
        header = table["header"]
        rows = table["rows"]

        # Insert empty rows at random positions
        for _ in range(self.times):
            empty_row = [""] * len(
                header
            )  # Create an empty row with the same number of columns
            insert_pos = random.randint(
                0, len(rows)
            )  # Get a random position to insert the empty row created
            rows.insert(insert_pos, empty_row)

        # Return the modified table
        return {"header": header, "rows": rows}


class MaskColumnsNames(TypeDependentAugmentor):
    """Mask the names of tables columns with dummies "Col1", "Col2" etc."""

    augmented_type = Table

    def process_value(self, table: Any) -> Any:
        masked_header = ["Col" + str(ind + 1) for ind in range(len(table["header"]))]

        return {"header": masked_header, "rows": table["rows"]}


class ShuffleColumnsNames(TypeDependentAugmentor):
    """Shuffle table columns names to be displayed in random order."""

    augmented_type = Table

    def process_value(self, table: Any) -> Any:
        shuffled_header = table["header"]
        random.shuffle(shuffled_header)

        return {"header": shuffled_header, "rows": table["rows"]}
