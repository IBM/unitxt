import re
import shutil

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


def construct_dict_str(d, indent=0, indent_delta=4, max_chars=None):
    """Constructs a formatted string of a dictionary.

    Args:
        d (dict): The dictionary to be formatted.
        indent (int, optional): The current level of indentation. Defaults to 0.
        indent_delta (int, optional): The amount of spaces to add for each level of indentation. Defaults to 4.
        max_chars (int, optional): The maximum number of characters for each line. Defaults to terminal width - 10.
    """
    max_chars = max_chars or shutil.get_terminal_size()[0] - 10
    indent_str = " " * indent
    indent_delta_str = " " * indent_delta
    res = ""

    for key, value in d.items():
        if isinstance(value, dict):
            res += f"{indent_str}{key}:\n"
            res += construct_dict_str(value, indent + indent_delta, max_chars=max_chars)
        else:
            str_value = str(value)
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


def print_dict(d, indent=0, indent_delta=4, max_chars=None):
    dict_str = construct_dict_str(d, indent, indent_delta, max_chars)
    dict_str = "\n" + dict_str
    logger.info(dict_str)


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
