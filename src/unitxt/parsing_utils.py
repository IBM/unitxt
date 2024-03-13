# An artifact is fetched from the catalog through the name by which it was added to that catalog.
# When first instantiated as an object (of class Artifact, or an extension thereof), values are set to the artifact's
# class arguments.
# When added to the catalog, the type (name of immediate class) along with the values of its arguments, are specified
# in the catalog record (json file) of that artifact, associated with the name given to that record upon adding to
# the catalog.
# When the artifact is later fetched from the catalog by that name, the values of its arguments are fetched as well,
# and used to instantiate the artifact into the object it was when added to the catalog.
#
# An addendum to this fetching process is a specification of alternative values to replace, upon fetching, (some of)
# the fetched values of the artifact's arguments, and to use the thus updated values for the instantiation following
# this fetching.
#
# The alternative arguments values, aka overwrites, are expressed as a string-ed series of key-value pairs, enclosed
# in square brackets, appended to the name of the artifact name.
#
# Overall, the formal definition of a query string, by which an artifact is fetched and instantiated, is as follows:
#
# query -> name | name [overwrites]
# overwrites -> assignment (, assignment)*
# assignment -> name = term
# term -> [ term (, term)* ] | { assignment (, assignment)* } | name_value | query
#
# name_value starting at a given point in the query string is the longest substring of the query string,
# that starts at that point, and ends upon reaching the end of the query string, or one of these chars:  [],:{}=
# spaces are allowed.
# name is a name_value that is not evaluated to an int, or float, or boolean.
#
# The following code processes a given query string, verifies that it conforms with the above format syntax, throwing
# exceptions otherwise, and returns a pair of: (a) artifact name in the catalog, and (b) a (potentially empty)
# dictionary whose keys are names of (some of) the class arguments of the artifact, associated with the alternative
# values to set to these arguments, upon the instantiation of the artifact as a response to this query.
#
# Note: the code does not verify that a name of an artifact's argument is indeed a name of an argument of that
# artifact. The instantiation process that follows the parsing will verify that.
# Also, if an alternative value of an argument is specified, in turn, as a query with overwrites, the conforming
# of that query with the above syntax is done when processing the major query of the artifact, but the parsing of the
# overwrites (in the argument's query) is delayed to the stage when the recursive instantiation of the major artifact
# reaches the instantiation of that argument.
#
#
from typing import Any, Tuple


def consume_name_val(instring: str) -> Tuple[Any, str]:
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


def consume_name(instring: str) -> Tuple[Any, str]:
    orig_instring = instring
    (name_val, instring) = consume_name_val(instring)
    if (
        name_val is None
        or isinstance(name_val, (int, float, bool))
        or len(name_val) == 0
    ):
        raise ValueError(f"malformed name at the beginning of: {orig_instring}")
    return (name_val, instring)


# flake8: noqa: C901
def consume_term(instring: str) -> Tuple[Any, str]:
    orig_instring = instring
    if instring.startswith("["):
        toret = []
        instring = instring[1:].strip()
        (term, instring) = consume_term(instring)
        toret.append(term)
        while instring.startswith(","):
            (term, instring) = consume_term(instring[1:].strip())
            toret.append(term)
        if not instring.startswith("]"):
            raise ValueError(f"malformed list in: {orig_instring}")
        instring = instring[1:].strip()
        return (toret, instring)

    if instring.startswith("{"):
        instring = instring[1:].strip()
        (items, instring) = consume_overwrites(instring, "}")
        if not instring.startswith("}"):
            raise ValueError(f"malformed dict in: {orig_instring}")
        instring = instring[1:].strip()
        return (items, instring)

    (name_val, instring) = consume_name_val(instring)
    instring = instring.strip()
    if not (
        name_val is None
        or isinstance(name_val, (int, float, bool))
        or len(name_val) == 0
    ) and instring.startswith("["):
        # term is a query with args
        (overwrites, instring) = consume_overwrites(instring[1:].strip(), "]")
        instring = instring.strip()
        if not instring.startswith("]"):
            raise ValueError(f"malformed query as a term in: {orig_instring}")
        instring = instring[1:].strip()
        toret = orig_instring[: len(orig_instring) - len(instring)]
        # argument's alternative value specified by query with overwrites.
        # the parsing of that query is delayed, to be synchronizes with the recursive loading
        # of the artifact from the catalog
        return (toret, instring)

    return (name_val, instring)


def consume_assignment(instring: str) -> Tuple[Any, str]:
    orig_instring = instring
    (name, instring) = consume_name(instring)

    if not instring.startswith("="):
        raise ValueError(f"malformed assignment in: {orig_instring}")
    (term, instring) = consume_term(instring[1:].strip())
    if (term is None) or not (isinstance(term, (int, float, bool)) or len(term) > 0):
        raise ValueError(f"malformed assigned value in: {orig_instring}")
    return ({name: term}, instring)


def consume_overwrites(instring: str, valid_follower: str) -> Tuple[Any, str]:
    if instring.startswith(valid_follower):
        return ({}, instring)
    (toret, instring) = consume_assignment(instring.strip())
    while instring.startswith(","):
        instring = instring[1:].strip()
        (assignment, instring) = consume_assignment(instring.strip())
        toret = {**toret, **assignment}
    return (toret, instring)


def consume_query(instring: str) -> Tuple[Tuple[str, any], str]:
    orig_instring = instring
    (name, instring) = consume_name(instring)
    instring = instring.strip()
    if len(instring) == 0 or not instring.startswith("["):
        return ((name, None), instring)

    (overwrites, instring) = consume_overwrites(instring[1:], "]")
    instring = instring.strip()
    if len(instring) == 0 or not instring.startswith("]"):
        raise ValueError(
            f"malformed end of query: overwrites not closed by ] in: {orig_instring}"
        )
    return ((name, overwrites), instring[1:].strip())


def parse_key_equals_value_string_to_dict(args: str) -> dict:
    """Parses a query string of the form 'key1=value1,key2=value2,...' into a dictionary.

    The function converts numeric values into integers or floats as appropriate, and raises an
    exception if the query string is malformed or does not conform to the expected format.

     :param query: The query string to be parsed.
     :return: A dictionary with keys and values extracted from the query string, with spaces stripped from keys.
    """
    instring = args.strip()
    if len(instring) == 0:
        return {}
    args_dict, instring = consume_overwrites(instring, " ")
    if len(instring.strip()) > 0:
        raise ValueError(f"Illegal key-values structure in {args}")
    return args_dict


def separate_inside_and_outside_square_brackets(s: str) -> Tuple[str, any]:
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

    instring = s.strip()
    orig_instring = instring
    if "[" not in instring:
        # no alternative values to artifact: consider the whole input string as an artifact name
        return (instring, None)
    if ("=" in instring and instring.find("=") < instring.find("[")) or (
        "," in instring and instring.find(",") < instring.find("[")
    ):
        # this could also constitute just overwrites for recipe's args, as conceived by fetch_artifact
        return (instring, None)
    # parse to identify artifact name and alternative values to artifact's arguments
    (query, instring) = consume_query(instring)
    if len(instring) > 0:
        raise ValueError(
            f"malformed end of query: excessive text following the ] that closes the overwrites in: '{orig_instring}'"
        )
    return query
