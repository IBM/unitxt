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


# Formal definition of query:
#  query -> assignment(=) (, assignment(=))*
#  assignment(delimeter) -> name_value delimeter term
#  term -> name_value | name_value[query] | [ term (, term)* ] | { assignment(:) (, assignment(:))* }
#
# a boolean parameter, return_dict, maintains a nested query: a query (potentially, indirectly) within another query,
#  in the form of a string, so that the process of parsing cohere with the recursive process of fetching the artifact.


def consume_name_val(instring: str) -> tuple:
    name_val = ""
    for char in instring:
        if char in "[],:{}=":
            break
        name_val += char
    instring = instring[len(name_val) :].strip()
    name_val = name_val.strip()

    if name_val == "True":
        return (True, instring)
    if name_val == "False":
        return (False, instring)

    sign = 1
    if name_val.startswith("-"):
        sign = -1
        name_val = name_val[1:]
    if name_val.isdigit():
        return (sign * int(name_val), instring)
    if name_val.replace(".", "", 1).isdigit() and name_val.count(".") < 2:
        return (sign * float(name_val), instring)

    if sign == -1:
        name_val = "-" + name_val
    return (name_val, instring)


# flake8: noqa: C901
def consume_term(instring: str, return_dict: bool) -> tuple:
    orig_instring = instring
    if instring.startswith("["):
        toret = []
        instring = instring[1:].strip()
        (term, instring) = consume_term(instring, return_dict)
        toret.append(term)
        while instring.startswith(","):
            (term, instring) = consume_term(instring[1:].strip(), return_dict)
            toret.append(term)
        if not instring.startswith("]"):
            raise ValueError(f"malformed list in: {orig_instring}")
        instring = instring[1:].strip()
        if not return_dict:
            toret = orig_instring[: len(orig_instring) - len(instring)]
        return (toret, instring)

    if instring.startswith("{"):
        instring = instring[1:].strip()
        (assignment, instring) = consume_assignment(instring, return_dict, ":")
        toret = assignment
        while instring.startswith(","):
            (assignment, instring) = consume_assignment(
                instring[1:].strip(), return_dict, ":"
            )
            if return_dict:
                toret = {**toret, **assignment}
        if not instring.startswith("}"):
            raise ValueError(f"malformed dict in: {orig_instring}")
        instring = instring[1:].strip()
        if not return_dict:
            toret = orig_instring[: len(orig_instring) - len(instring)]
        return (toret, instring)

    (name_val, instring) = consume_name_val(instring)
    if instring.startswith("["):
        (quey, instring) = consume_query(instring[1:].strip(), False)
        if not instring.startswith("]"):
            raise ValueError(f"malformed query in: {orig_instring}")
        instring = instring[1:].strip()
        toret = orig_instring[: len(orig_instring) - len(instring)]
        return (toret, instring)
    return (name_val, instring)


def consume_assignment(instring: str, return_dict: bool, delimeter: str) -> tuple:
    orig_instring = instring
    (name_val, instring) = consume_name_val(instring)
    if (
        name_val is None
        or isinstance(name_val, (int, float, bool))
        or len(name_val) == 0
    ):
        raise ValueError(f"malformed key in assignment that starts: {orig_instring}")
    if not instring.startswith(delimeter):
        raise ValueError(f"malformed assignment in: {orig_instring}")
    (term, instring) = consume_term(instring[1:].strip(), return_dict)
    if (term is None) or not (isinstance(term, (int, float, bool)) or len(term) > 0):
        raise ValueError(f"malformed assignment in: {orig_instring}")
    if return_dict:
        return ({name_val: term}, instring)
    toret = orig_instring[: len(orig_instring) - len(instring)]
    return (toret, instring)


def consume_query(instring: str, return_dict: bool) -> tuple:
    orig_instring = instring
    (toret, instring) = consume_assignment(instring.strip(), return_dict, "=")
    while instring.startswith(","):
        instring = instring[1:].strip()
        (assignment, instring) = consume_assignment(instring.strip(), return_dict, "=")
        if return_dict:
            toret = {**toret, **assignment}
        else:
            toret = orig_instring[: len(orig_instring) - len(instring)]
    return (toret, instring)


def parse_key_equals_value_string_to_dict(query: str) -> dict:
    """Parses a query string of the form 'key1=value1,key2=value2,...' into a dictionary.

    The function converts numeric values into integers or floats as appropriate, and raises an
    exception if the query string is malformed or does not conform to the expected format.

     :param query: The query string to be parsed.
     :return: A dictionary with keys and values extracted from the query string, with spaces stripped from keys.
    """
    instring = query
    qu, instring = consume_query(instring, True)
    if len(instring.strip()) > 0:
        raise ValueError(f"Illegal query structure in {query}")
    return qu
