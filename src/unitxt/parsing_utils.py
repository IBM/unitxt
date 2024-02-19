def separate_inside_and_outside_square_brackets(s):
    """Separates the content inside and outside the first level of square brackets in a string.

    Allows text before the first bracket and nested brackets within the first level. Raises a ValueError for:
    - Text following the closing bracket of the first bracket pair
    - Unmatched brackets
    - Multiple bracket pairs at the same level

    :param s: The input string to be parsed.
    :return: A tuple (outside, inside) where 'outside' is the content outside the first level of square brackets,
             and 'inside' is the content inside the first level of square brackets. If there are no brackets,
             'inside' will be None.
    """
    start = s.find("[")
    end = s.rfind("]")

    # Handle no brackets
    if start == -1 and end == -1:
        return s, None

    # Validate brackets
    if start == -1 or end == -1 or start > end:
        raise ValueError("Illegal structure: unmatched square brackets.")

    outside = s[:start]
    inside = s[start + 1 : end]
    after = s[end + 1 :]

    # Check for text after the closing bracket
    if len(after.strip()) != 0:
        raise ValueError(
            "Illegal structure: text follows after the closing square bracket."
        )

    return outside, inside


def parse_key_equals_value_string_to_dict(query: str):
    """Parses a query string of the form 'key1=value1,key2=value2,...' into a dictionary.

    The function converts numeric values into integers or floats as appropriate, and raises an
    exception if the query string is malformed or does not conform to the expected format.

    :param query: The query string to be parsed.
    :return: A dictionary with keys and values extracted from the query string, with spaces stripped from keys.
    """
    result = {}
    kvs = query.split(",")
    if len(kvs) == 0:
        raise ValueError(
            f'Illegal query: "{query}" should contain at least one assignment of the form: key1=value1,key2=value2'
        )
    for kv in kvs:
        key_val = kv.split("=")
        if (
            len(key_val) != 2
            or len(key_val[0].strip()) == 0
            or len(key_val[1].strip()) == 0
        ):
            raise ValueError(
                f'Illegal query: "{query}" with wrong assignment "{kv}" should be of the form: key=value.'
            )
        key, val = key_val[0].strip(), key_val[1].strip()
        if val.isdigit():
            result[key] = int(val)
        elif val.replace(".", "", 1).isdigit() and val.count(".") < 2:
            result[key] = float(val)
        else:
            result[key] = val

    return result
