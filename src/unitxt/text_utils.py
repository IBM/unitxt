import re
import shutil


def split_words(s):
    """
    Splits a string into words based on PascalCase, camelCase, snake_case,
    kebab-case, and numbers attached to strings.

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
    words = s.split()
    return words


def is_camel_case(s):
    """
    Checks if a string is in camelCase.

    Args:
        s (str): The string to be checked.

    Returns:
        bool: True if the string is in camelCase, False otherwise.
    """
    return re.match(r"^[A-Z]+([a-z0-9]*[A-Z]*[a-z0-9]*)*$", s) is not None


def is_snake_case(s):
    """
    Checks if a string is in snake_case.

    Args:
        s (str): The string to be checked.

    Returns:
        bool: True if the string is in snake_case, False otherwise.
    """
    return re.match(r"^[a-z0-9]+(_[a-z0-9]+)*$", s) is not None


def camel_to_snake_case(s):
    """
    Converts a string from camelCase to snake_case.

    Args:
        s (str): The string to be converted.

    Returns:
        str: The string converted to snake_case.
    """
    # Add an underscore before every uppercase letter that is followed by a lowercase letter or digit and not preceded by an underscore, a hyphen or an uppercase letter
    s = re.sub(r"(?<=[^A-Z_-])([A-Z])", r"_\1", s)

    # Ensure there's an underscore before any uppercase letter that's followed by a lowercase letter or digit and comes after a sequence of uppercase letters
    s = re.sub(r"([A-Z]+)([A-Z][a-z0-9])", r"\1_\2", s)

    s = s.lower()
    return s


def print_dict(d, indent=0, indent_delta=4, max_chars=None):
    """
    Prints a dictionary in a formatted manner, taking into account the terminal
    width.

    Args:
        d (dict): The dictionary to be printed.
        indent (int, optional): The current level of indentation. Defaults to 0.
        indent_delta (int, optional): The amount of spaces to add for each level of indentation. Defaults to 4.
        max_chars (int, optional): The maximum number of characters for each line. Defaults to terminal width - 10.
    """
    max_chars = (
        max_chars or shutil.get_terminal_size()[0] - 10
    )  # Get terminal size if max_chars not set
    indent_str = " " * indent
    indent_delta_str = " " * indent_delta

    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_dict(value, indent=indent + indent_delta, max_chars=max_chars)
        else:
            # Value is not a dict, print as a string
            str_value = str(value)

            line_width = max_chars - indent
            # Split value by newline characters and handle each line separately
            lines = str_value.split("\n")
            print(f"{indent_str}{key}:")
            for line in lines:
                if len(line) + len(indent_str) + indent_delta > line_width:
                    # Split long lines into multiple lines
                    print(f"{indent_str}{indent_delta_str}{line[:line_width]}")
                    for i in range(line_width, len(line), line_width):
                        print(f"{indent_str}{indent_delta_str}{line[i:i+line_width]}")
                else:
                    print(f"{indent_str}{indent_delta_str}{line}")
                key = ""  # Empty the key for lines after the first one


def nested_tuple_to_string(nested_tuple: tuple) -> str:
    """
    Converts a nested tuple to a string, with elements separated by
    underscores.

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
