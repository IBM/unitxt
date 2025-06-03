import logging
import textwrap
from typing import List, Optional

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def print_generic_table(headers: list, data: List, col_widths: Optional[dict] = None):
    """Prints a table with the given headers, column widths (with a default uniform width), and data.

    Args:
        headers (list): A list of column headers.
        col_widths (dict, optional): A dictionary with column names as keys and their respective widths as values.
                                     If not provided, all columns will have a uniform width of 20.
        data (list): A list of dictionaries, where each dictionary represents a row with column names as keys.
    """
    # Set default uniform width if col_widths is not provided
    if col_widths is None:
        col_widths = {header: 20 for header in headers}

    # Calculate the total table width based on column widths
    table_width = (
        sum(col_widths.values()) + len(col_widths) * 3 + 2
    )  # Adjust for separators and | at left and right

    # Print separator before the table
    logger.info("=" * table_width)

    # Create the header row
    header_row = " | ".join([f"{header:<{col_widths[header]}}" for header in headers])

    # logger.info the header
    logger.info(f"| {header_row} |")
    logger.info("=" * table_width)  # Separator line after the header

    # Loop through the data and print each row
    for row in data:
        # Wrap text to fit within column widths and prepare wrapped rows
        wrapped_columns = {
            col: textwrap.fill(str(row[col]), width=col_widths[col]) for col in headers
        }

        # Split wrapped columns into multiple lines
        wrapped_lines = {col: wrapped_columns[col].split("\n") for col in headers}

        # Find the maximum number of lines across all columns
        max_lines = max(len(wrapped_lines[col]) for col in headers)

        # Print each line
        for i in range(max_lines):
            line = (
                "| "
                + " | ".join(
                    [
                        f"{wrapped_lines[col][i] if i < len(wrapped_lines[col]) else '':<{col_widths[col]}}"
                        for col in headers
                    ]
                )
                + " |"
            )
            logger.info(line)
        logger.info("=" * table_width)  # Separator line after each row
