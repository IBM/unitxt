import re
import shutil
from typing import List, Tuple

from .logging_utils import get_logger

logger = get_logger()


def split_words(s):
    """Splits a string into words based on PascalCase, camelCase, snake_case, kebab-case, and numbers attached to strings.

    Args:
        s (str): The string to be split.

    Returns:
        list: The list of words obtained after splitting the string.
    """
    # Split PascalCase or camelCase
    s = re.sub(r"([A-Z][a-z]+)", r" \1", re.sub(r"([A-Z]+)", r" \1", s)).strip()
    # Split snake_case or kebab-case
    s = re.sub(r"[_-]", " ", s)
    # Split numbers attached to strings
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)
    # Split the string into words based on spaces
    return s.split()


def is_camel_case(s):
    """Checks if a string is in camelCase.

    Args:
        s (str): The string to be checked.

    Returns:
        bool: True if the string is in camelCase, False otherwise.
    """
    return re.match(r"^[A-Z]+([a-z0-9]*[A-Z]*[a-z0-9]*)*$", s) is not None


def is_snake_case(s):
    """Checks if a string is in snake_case.

    Args:
        s (str): The string to be checked.

    Returns:
        bool: True if the string is in snake_case, False otherwise.
    """
    return re.match(r"^[a-z0-9]+(_[a-z0-9]+)*$", s) is not None


def camel_to_snake_case(s):
    """Converts a string from camelCase to snake_case.

    Args:
        s (str): The string to be converted.

    Returns:
        str: The string converted to snake_case.
    """
    # Add an underscore before every uppercase letter that is followed by a lowercase letter or digit and not preceded by an underscore, a hyphen or an uppercase letter
    s = re.sub(r"(?<=[^A-Z_-])([A-Z])", r"_\1", s)

    # Ensure there's an underscore before any uppercase letter that's followed by a lowercase letter or digit and comes after a sequence of uppercase letters
    s = re.sub(r"([A-Z]+)([A-Z][a-z0-9])", r"\1_\2", s)

    return s.lower()


def construct_dict_str(d, indent=0, indent_delta=4, max_chars=None, keys=None):
    """Constructs a formatted string of a dictionary.

    Args:
        d (dict): The dictionary to be formatted.
        indent (int, optional): The current level of indentation. Defaults to 0.
        indent_delta (int, optional): The amount of spaces to add for each level of indentation. Defaults to 4.
        max_chars (int, optional): The maximum number of characters for each line. Defaults to terminal width - 10.
        keys (List[Str], optional): the list of fields to print
    """
    max_chars = max_chars or shutil.get_terminal_size()[0] - 10
    indent_str = " " * indent
    indent_delta_str = " " * indent_delta
    res = ""

    if keys is None:
        keys = d.keys()
    for key in keys:
        if key not in d.keys():
            raise ValueError(
                f"Dictionary does not contain field {key} specified in 'keys' argument. The available keys are {d.keys()}"
            )
        value = d[key]
        if isinstance(value, dict):
            res += f"{indent_str}{key}:\n"
            res += construct_dict_str(value, indent + indent_delta, max_chars=max_chars)
        else:
            str_value = str(value)
            str_value = re.sub(r"\w+=None, ", "", str_value)
            str_value = re.sub(r"\w+={}, ", "", str_value)
            str_value = re.sub(r"\w+=\[\], ", "", str_value)
            line_width = max_chars - indent
            lines = str_value.split("\n")
            res += f"{indent_str}{key} ({type(value).__name__}):\n"
            for line in lines:
                if len(line) + len(indent_str) + indent_delta > line_width:
                    res += f"{indent_str}{indent_delta_str}{line[:line_width]}\n"
                    for i in range(line_width, len(line), line_width):
                        res += f"{indent_str}{indent_delta_str}{line[i:i+line_width]}\n"
                else:
                    res += f"{indent_str}{indent_delta_str}{line}\n"
                key = ""  # Empty the key for lines after the first one
    return res


def print_dict(
    d, indent=0, indent_delta=4, max_chars=None, keys_to_print=None, log_level="info"
):
    dict_str = construct_dict_str(d, indent, indent_delta, max_chars, keys_to_print)
    dict_str = "\n" + dict_str
    getattr(logger, log_level)(dict_str)


def nested_tuple_to_string(nested_tuple: tuple) -> str:
    """Converts a nested tuple to a string, with elements separated by underscores.

    Args:
        nested_tuple (tuple): The nested tuple to be converted.

    Returns:
        str: The string representation of the nested tuple.
    """
    result = []
    for item in nested_tuple:
        if isinstance(item, tuple):
            result.append(nested_tuple_to_string(item))
        else:
            result.append(str(item))
    return "_".join(result)


def is_made_of_sub_strings(string, sub_strings):
    pattern = "^(" + "|".join(map(re.escape, sub_strings)) + ")+$"
    return bool(re.match(pattern, string))


# Giveמ all the lines of a card preparer file, e.g. all the lines of prepare/cards/cohere_for_ai.py,
# and an object name, e.g. TaskCard(,
# return the ordinal number of the line that starts that object, in our example: the
# line number of the following line (notice that the line where TaskCard is imported
# is not supposed to return):
#         card = TaskCard(
# and the line number of the line that ends the object, in our case the line that include
# the matching close:
#         )
# This util depends on ruff to ensure this setting of the card file: that a close of one
# tag and the open of the next tag, do not sit in same line, when both tags being
# major level within TaskCard.
# It also prepares for the case that  __description__ tag does not contain balanced
# parentheses, since it is often cut in the middle, (with  "... see more at")
# flake8: noqa: B007
# flake8: noqa: C901
def lines_defining_obj_in_card(
    all_lines: List[str], obj_name: str, start_search_at_line: int = 0
) -> Tuple[int, int]:
    for starting_line in range(start_search_at_line, len(all_lines)):
        line = all_lines[starting_line]
        if obj_name in line:
            break
    if obj_name not in line:
        # obj_name found no where in the input lines
        return (-1, -1)
    num_of_opens = 0
    num_of_closes = 0
    ending_line = starting_line - 1
    while ending_line < len(all_lines):
        ending_line += 1

        if "__description__" in all_lines[ending_line]:
            # can not trust parentheses inside description, because this is mainly truncated
            # free text.
            # We do trust the indentation enforced by ruff, and the way we build __description__:
            # a line consisting of only __description__=(
            # followed by one or more lines of text, can not trust opens and closes
            # in them, followed by a line consisting of only:  ),
            # where the ) is indented with the beginning of __description__
            # We also prepare for the case that, when not entered by us, __description__=
            # is not followed by a ( and the whole description does not end with a single ) in its line.
            # We build on ruff making the line following the description start with same indentation
            # or 4 less (i.e., the following line is the closing of the card).
            tag_indentation = all_lines[ending_line].index("__description__")
            starts_with_parent = "__description__=(" in all_lines[ending_line]
            if starts_with_parent:
                last_line_to_start_with = (" " * tag_indentation) + r"\)"
            else:
                # actually, the line that follows the description
                last_line_to_start_with1 = (" " * tag_indentation) + "[^ ]"
                last_line_to_start_with2 = (" " * (tag_indentation - 4)) + "[^ ]"
                last_line_to_start_with = (
                    "("
                    + last_line_to_start_with1
                    + "|"
                    + last_line_to_start_with2
                    + ")"
                )
            ending_line += 1
            while not re.search("^" + last_line_to_start_with, all_lines[ending_line]):
                ending_line += 1
            if "__description__" in obj_name:
                return (
                    starting_line,
                    ending_line if starts_with_parent else ending_line - 1,
                )

            if starts_with_parent:
                ending_line += 1

            # we conrinue in card, having passed the description, ending line points
            # to the line that follows description

        num_of_opens += len(re.findall(r"[({[]", all_lines[ending_line]))
        num_of_closes += len(re.findall(r"[)}\]]", all_lines[ending_line]))
        if num_of_closes == num_of_opens:
            break

    if num_of_closes != num_of_opens:
        raise ValueError(
            "input lines were exhausted before the matching close is found"
        )

    return (starting_line, ending_line)
