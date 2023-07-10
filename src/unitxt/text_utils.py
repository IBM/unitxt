import re


def split_words(s):
    # Split PascalCase or camelCase
    s = re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", s)).strip()
    # Split snake_case or kebab-case
    s = re.sub("[_-]", " ", s)
    # Split numbers attached to strings
    s = re.sub("([a-zA-Z])(\d)", r"\1 \2", s)
    s = re.sub("(\d)([a-zA-Z])", r"\1 \2", s)
    # Split the string into words based on spaces
    words = s.split()
    return words


def is_camel_case(s):
    # The string must start with an uppercase letter, followed by zero or more sequences of an uppercase letter followed by zero or more lowercase letters.
    return re.match(r"^[A-Z]+([a-z0-9]*[A-Z]*[a-z0-9]*)*$", s) is not None


def is_snake_case(s):
    # The string must start with a lowercase letter, followed by zero or more sequences of an underscore followed by one or more lowercase letters.
    return re.match(r"^[a-z0-9]+(_[a-z0-9]+)*$", s) is not None


def camel_to_snake_case(s):
    # Add an underscore before every uppercase letter that is followed by a lowercase letter or digit and not preceded by an underscore, a hyphen or an uppercase letter
    s = re.sub("(?<=[^A-Z_-])([A-Z])", r"_\1", s)

    # Ensure there's an underscore before any uppercase letter that's followed by a lowercase letter or digit and comes after a sequence of uppercase letters
    s = re.sub("([A-Z]+)([A-Z][a-z0-9])", r"\1_\2", s)

    s = s.lower()
    return s


import shutil


def print_dict(d, indent=0, indent_delta=4, max_chars=None):
    max_chars = max_chars or shutil.get_terminal_size()[0] - 10  # Get terminal size if max_chars not set
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
    result = []
    for item in nested_tuple:
        if isinstance(item, tuple):
            result.append(nested_tuple_to_string(item))
        else:
            result.append(str(item))
    return "_".join(result)


if __name__ == "__main__":
    # Define test cases
    test_cases = [
        ("example1", ["example", "1"]),
        ("exampleOne", ["example", "One"]),
        ("123example456", ["123", "example", "456"]),
        ("happyDay", ["happy", "Day"]),
        ("thisIsATest", ["this", "Is", "A", "Test"]),
        ("TestAI2023", ["Test", "AI", "2023"]),
        ("stringWith1Number", ["string", "With", "1", "Number"]),
        ("camelCaseExample", ["camel", "Case", "Example"]),
        ("snake_case_example", ["snake", "case", "example"]),
        ("snake_case2example3", ["snake", "case", "2", "example", "3"]),
        ("kebab-case-example", ["kebab", "case", "example"]),
        ("kebab-case2example3", ["kebab", "case", "2", "example", "3"]),
        ("PascalCaseExample", ["Pascal", "Case", "Example"]),
        ("Title Case Example", ["Title", "Case", "Example"]),
        ("Mixed1Example_case", ["Mixed", "1", "Example", "case"]),
        ("Mixed2Example-case", ["Mixed", "2", "Example", "case"]),
        ("Mixed3_Example-case", ["Mixed", "3", "Example", "case"]),
        ("UPPERCASEEXAMPLE", ["UPPERCASEEXAMPLE"]),
        ("lowercaseexample", ["lowercaseexample"]),
        ("mixedUPanddown", ["mixed", "U", "Panddown"]),
    ]

    # Loop through test cases
    for i, (input_string, expected_output) in enumerate(test_cases, 1):
        # Apply function and check result
        if split_words(input_string) != expected_output:
            print(f"Failed on example {i}: {input_string}")
            print(f"Expected: {expected_output}, but got: {split_words(input_string)}\n")

    is_camel_case_test_cases = [
        ("isCamelCase", False),
        ("notCamelCase", False),
        ("camelCase", False),
        ("Notcamelcase", True),
        ("camel_Case", False),
        ("camelCase123", False),
        ("camelcase", False),
        ("CAMELCASE", True),
        ("camel-case", False),
        ("HFLoader", True),
    ]

    for input_string, expected_output in is_camel_case_test_cases:
        if is_camel_case(input_string) != expected_output:
            print(f"Failed on is_camel_case: {input_string}")
            print(f"Expected: {expected_output}, but got: {is_camel_case(input_string)}\n")

    is_snake_case_test_cases = [
        ("is_snake_case", True),
        ("Not_snake_case", False),
        ("snake_case", True),
        ("snake_Case", False),
        ("Snakecase", False),
        ("snake-case", False),
        ("snake_case123", True),
        ("123snake_case", True),
        ("snakecase", True),
    ]

    for input_string, expected_output in is_snake_case_test_cases:
        if is_snake_case(input_string) != expected_output:
            print(f"Failed on is_snake_case: {input_string}")
            print(f"Expected: {expected_output}, but got: {is_snake_case(input_string)}\n")

    camel_to_snake_case_test_cases = [
        ("camelToSnake", "camel_to_snake"),
        ("CamelToSnake", "camel_to_snake"),
        ("CamelToSnakeCase", "camel_to_snake_case"),
        ("camelToSnakeCase123", "camel_to_snake_case123"),
        ("123CamelToSnakeCase", "123_camel_to_snake_case"),
        ("camelTo_Snake_Case", "camel_to__snake__case"),
        ("camelTo-Snake-Case", "camel_to-_snake-_case"),
        ("camelToSnakeCASE", "camel_to_snake_case"),
        ("CAMELToSnakeCase", "camel_to_snake_case"),
        ("camelToSNAKECase", "camel_to_snake_case"),
        ("HFLoader", "hf_loader"),
    ]

    for input_string, expected_output in camel_to_snake_case_test_cases:
        if camel_to_snake_case(input_string) != expected_output:
            print(f"Failed on camel_to_snake_case: {input_string}")
            print(f"Expected: {expected_output}, but got: {camel_to_snake_case(input_string)}\n")
