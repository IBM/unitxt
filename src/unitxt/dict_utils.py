import re
from typing import Any, List, Tuple

from .text_utils import construct_dict_str

indx = re.compile(r"^(\d+)$")


def is_index(string):
    return bool(indx.match(string))


name = re.compile(r"^[\w. -]+$")


def is_name(string):
    return bool(name.match(string))


def is_wildcard(string):
    return string == "*"


# formal definition of qpath syntax by which a query is specified:
# qpath -> A (/A)*
# A -> name | * | non-neg-int
# name -> a string satisfying is_name above.
#
# A path p in dictionary dic, leading to element (aka subfield) el, is said to match query qpath
# (alternatively said: query qpath matches path p in dic),
# if the following recursively defined condition is satisfied:
# (1) the prefix of length 0 of qpath (i.e., pref = "") matches the empty path in dic, the path leading to the whole of dic.
# (2) Denoting by el the element in dic lead to by the path in dic that matches the prefix pref of qpath
# (el must be a list or dictionary, since led to by a path matching a prefix of qpath, and not the whole of qpath),
# and by A (as the definition above) the component, DIFFERENT from *, in qpath, that follows pref, the element
# lead to by the path in dic matching query pref/A is el[A]. If el[A] is missing from dic, then no path in dic matches
# pref/A, that is either a longer prefix of qpath, or the whole of qpath,
# and hence no path in dic matches query qpath. (E.g., when el is a list, A must match indx, and its
# int value should be smaller than len(el) in order for the path in dic leading to element el[A] to match pref/A)
# (3) Denoting as in (2), now with A == * : when el is a list, each and every element in the set:
# {el[0], el[1], .. , el[len(el)-1]} is said to be lead to by a path matching pref/*
# and when el is a dict, each and every element in the set {el[k] for k being a key in el} is said to be lead
# to by a path matching pref/*
#
# An element el lead to by path p that matches qpath as a whole is thus either a list member (when indx.match the last
# component of p, indexing into el) or a dictionary item (the key of which equals the last component of p). The value
# of el (i.e. el[last component of p]) is returned (dic_get) or popped (dic_delete) or replaced by a new value (dic_set).
#
# Thus, for a query with no *, dic contains at most one element the path to which matches the query.
# If there is such one in dic - the function (either dict_get, dict_set, or dict_delete) operates on
# that element according to its arguments, other than not_exist_ok.
# If there is not any such element in dic - the function throws or does not throw an exception, depending
# on flag not_exist_ok.
# For a query with *, there could be up to as many as there are values to match the *
# (i.e., length of the list element or dictionary element the children of which match the *; and
# for more than one * in the query -- this effect multiplies)
# Each of the three functions below (dict_get, dict_set, dict_delete) applies the requested
# operation (read, set, or delete) to each and every element el in dic, the path to which matches the query in whole,
# and reads a value from, or sets a new value to, or pops the value out from dic.
#
# If no path in dic matches the query, then if not_exist_ok=False, the function throws an exception;
# but if not_exist_ok=True, the function returns a default value (dict_get) or does nothing (dict_delete)
# or generates the needed missing suffix, when consists of at most one component (dict_set, see details below).
#
# Each of the functions below scans the dic-tree recursively.
# It swallows all exceptions, in order to not stop prematurely, before the scan
# has exhaustively found each and every matching path.


# validate and normalizes into components
def validate_query_and_break_to_components(query: str) -> List[str]:
    if not isinstance(query, str) or len(query) == 0:
        raise ValueError(
            f"invalid query: either not a string or an empty string: {query}"
        )
    query = query.replace("//", "/").strip()
    if query.startswith("/"):
        query = query[1:]
        # ignore the leading /, all paths are treated as coming from root of dic

    if query.endswith("/"):
        query = query + "*"
        # same meaning, and make sure the / is not lost when splitting

    components = query.split("/")
    components = [component.strip() for component in components]
    for component in components:
        if not (is_name(component) or is_wildcard(component)):
            raise ValueError(
                f"Component {component} in input query is none of: valid field-name, non-neg-int, or '*'"
            )
    return components


def is_subpath(subpath, fullpath):
    # Split the paths into individual components
    subpath_components = validate_query_and_break_to_components(subpath)
    fullpath_components = validate_query_and_break_to_components(fullpath)

    # Check if the full path starts with the subpath
    return fullpath_components[: len(subpath_components)] == subpath_components


# We are on current_element, going down from it via query[index_into_query].
# query comprising at least two components is assumed. dic_delete worries about
# single component queries without invoking qpath_delete.
# user wants to delete the element led to by the whole query from the element led to
# by the prefix of the query consisting of all but the last component.
# Returned value is a pair (boolean, element_of_input dic or None)
# the first component signals whether the second is yielded from a reach to the query end,
# or rather -- a result of a failure before query end has been reached.
# If the first component is True, the second is current_element following a successful delete
def delete_values(
    current_element: Any,
    query: List[str],
    index_into_query: int,
    remove_empty_ancestors=False,
) -> Tuple[bool, Any]:
    component = query[index_into_query]
    if index_into_query == -1:
        # need to delete a subelement, identified by component, of current_element.
        if is_wildcard(component):
            # delete all members of the list or dict
            current_element = [] if isinstance(current_element, list) else {}
            return (True, current_element)
        # component is a either a dictionary key or an index into a list,
        # pop the respective element from current_element
        if is_index(component) and isinstance(current_element, list):
            component = int(component)
        try:
            current_element.pop(component)
            return (True, current_element)
        except:
            # no continuation in dic, from current_element down, that matches the query
            return (False, None)

    # index_into_query < -1
    if component == "*":
        # current_element must be a dict or list. We need to update value for all its members
        # through which passes a path that reached the query end
        if isinstance(current_element, dict):
            key_values = list(current_element.items())
            keys, values = zip(*key_values)
        elif isinstance(current_element, list):
            keys = list(range(len(current_element)))
            values = current_element
        else:
            return (False, None)

        any_success = False
        for i in range(
            len(keys) - 1, -1, -1
        ):  # going backward to allow popping from a list
            try:
                success, new_val = delete_values(
                    current_element=values[i],
                    query=query,
                    index_into_query=index_into_query + 1,
                    remove_empty_ancestors=remove_empty_ancestors,
                )
                if not success:
                    continue
                any_success = True
                if (len(new_val) == 0) and remove_empty_ancestors:
                    current_element.pop(keys[i])
                else:
                    current_element[keys[i]] = new_val

            except:
                continue
        return (any_success, current_element)

    # current component is index into a list or a key into a dictionary
    if is_index(component) and isinstance(current_element, list):
        component = int(component)
    try:
        success, new_val = delete_values(
            current_element=current_element[component],
            query=query,
            index_into_query=index_into_query + 1,
            remove_empty_ancestors=remove_empty_ancestors,
        )
        if not success:
            return (False, None)
        if (len(new_val) == 0) and remove_empty_ancestors:
            current_element.pop(component)
        else:
            current_element[component] = new_val
        return (True, current_element)
    except:
        return (False, None)


def dict_delete(
    dic: dict,
    query: str,
    not_exist_ok: bool = False,
    remove_empty_ancestors=False,
):
    # We remove from dic the value from each and every element lead to by a path matching the query.
    # If remove_empty_ancestors=True, and the removal of any such value leaves its containing element (list or dict)
    # within dic empty -- remove that element as well, and continue recursively, but stop one step before deleting dic
    # altogether, even if became {}. If successful, changes dic into its new shape

    if dic is None or not isinstance(dic, (list, dict)):
        raise ValueError(
            f"dic {dic} is either None or not a list nor a dict. Can not delete from it."
        )

    if len(query) == 0:
        raise ValueError(
            "Query is an empty string, implying the deletion of dic as a whole. This can not be done via this function call."
        )

    if isinstance(dic, dict) and query.strip() in dic:
        dic.pop(query.strip())
        return

    qpath = validate_query_and_break_to_components(query)

    try:
        success, new_val = delete_values(
            current_element=dic,
            query=qpath,
            index_into_query=(-1) * len(qpath),
            remove_empty_ancestors=remove_empty_ancestors,
        )

        if success:
            if new_val == {}:
                dic.clear()
            return

        if not not_exist_ok:
            raise ValueError(
                f"An attempt to delete from dictionary {dic}, an element {query}, that does not exist in the dictionary, while not_exist_ok=False"
            )

    except Exception as e:
        raise ValueError(f"query {query} matches no path in dictionary {dic}") from e


# returns all the values sitting inside dic, in all the paths that match query_path
# if query includes * then return a list of values reached by all paths that match the query
# flake8: noqa: C901
def get_values(
    current_element: Any,
    query: List[str],
    index_into_query: int,
) -> Tuple[bool, Any]:
    # going down from current_element through query[index_into_query].
    if index_into_query == 0:
        return (True, current_element)

    # index_into_query < 0
    component = query[index_into_query]
    if component == "*":
        # current_element must be a list or a dictionary
        if not isinstance(current_element, (list, dict)):
            return (False, None)  # nothing good from here down the query
        to_ret = []
        if isinstance(current_element, dict):
            sub_elements = list(current_element.values())
        else:  # a list
            sub_elements = current_element
        for sub_element in sub_elements:
            try:
                success, val = get_values(
                    sub_element,
                    query,
                    index_into_query + 1,
                )
                if success:
                    to_ret.append(val)
            except:
                continue

        return (len(to_ret) > 0 or index_into_query == -1, to_ret)
        # when * is the last component, it refers to 'all the contents' of an empty list being current_element.
    # next_component is indx or name, current_element must be a list or a dict
    if is_index(component) and isinstance(current_element, (list, str)):
        # for backward compatibility, use the fact that also a string can be indexed into
        component = int(component)
    try:
        success, new_val = get_values(
            current_element[component],
            query,
            index_into_query + 1,
        )
        if success:
            return (True, new_val)
        return (False, None)
    except:
        return (False, None)


# going down from current_element via query[index_into_query]
# returns the updated current_element.
# if not_exist_ok and the last component is missing from dic -- generate that last component in that dic.
# But not any earier component. E.g., not that containing dict.
# That is, through processing the query, that most that can be added to the processed dic, is a field in a
# dictionary or an entry in a list (extending an existing list). If more is needed to add to dic, pass it
# through the value being set, which can be anything, structured and complex, or simple.
def set_values(
    current_element: Any,
    value: Any,
    index_into_query: int,
    query: List[str],
    not_exist_ok: bool,
    set_multiple: bool = False,
) -> Tuple[bool, Any]:
    if index_into_query == 0:
        return (True, value)  # matched query all along!

    component = query[index_into_query]
    if component == "*":
        if set_multiple:
            if isinstance(current_element, dict) and len(current_element) != len(value):
                return (False, None)
            if isinstance(current_element, list) and len(current_element) > len(value):
                return (False, None)
            if len(current_element) < len(value):
                if not not_exist_ok or index_into_query < -1:
                    return (False, None)
                # current_element must be a list, extend current_element to the length needed, but only
                # if at the last component of the query
                current_element.extend([None] * (len(value) - len(current_element)))
        if current_element == []:
            current_element = [None] * (
                len(value)
                if set_multiple
                else value is None
                or not isinstance(value, list)
                or len(value) > 0
                or index_into_query < -1
            )
        # now current_element is of size suiting value
        if isinstance(current_element, dict):
            keys = sorted(current_element.keys())
        else:
            keys = list(range(len(current_element)))

        any_success = False
        for i in range(len(keys)):
            try:
                success, new_val = set_values(
                    current_element=current_element[keys[i]],
                    value=value[i] if set_multiple else value,
                    index_into_query=index_into_query + 1,
                    query=query,
                    not_exist_ok=not_exist_ok,
                    set_multiple=False,  # now used, not allowed again,
                )
                if not success:
                    continue
                any_success = True
                current_element[keys[i]] = new_val

            except:
                continue
        return (
            any_success or (len(keys) == 0 and index_into_query == -1),
            current_element,
        )

    # component is an index into a list or a key into a dictionary
    if is_index(component) and isinstance(current_element, list):
        # current_element is a list
        component = int(component)
        if component >= len(current_element):
            if not not_exist_ok or index_into_query < -1:
                # preparing a new place for the value to set is only allowed at the end of the query
                return (False, None)
            # extend current_element to the length needed
            current_element.extend([None] * (component + 1 - len(current_element)))
        next_current_element = current_element[component]
    else:  # component is a key into a dictionary
        if not isinstance(current_element, dict):
            return (False, None)
        if component not in current_element:
            if not not_exist_ok or index_into_query < -1:
                # preparing a new place for the value to set is only allowed at the end of the query
                return (False, None)
        next_current_element = (
            None if component not in current_element else current_element[component]
        )
    try:
        success, new_val = set_values(
            current_element=next_current_element,
            value=value,
            index_into_query=index_into_query + 1,
            query=query,
            not_exist_ok=not_exist_ok,
            set_multiple=set_multiple,
        )
        if success:
            current_element[component] = new_val
            return (True, current_element)
        return (False, None)
    except:
        return (False, None)


# the returned values are ordered by the lexicographic order of the paths leading to them
def dict_get(
    dic: dict,
    query: str,
    not_exist_ok: bool = False,
    default: Any = None,
):
    if len(query.strip()) == 0:
        return dic

    if dic is None:
        raise ValueError("Can not get any value from a dic that is None")

    if isinstance(dic, dict) and query.strip() in dic:
        return dic[query.strip()]

    components = validate_query_and_break_to_components(query)
    if len(components) > 1:
        try:
            success, values = get_values(dic, components, -1 * len(components))
            if success:
                return values
        except Exception as e:
            raise ValueError(
                f"dict_get: query '{query}' did not match any item in dict:\n{construct_dict_str(dic)}"
            ) from e

        if not_exist_ok:
            return default

        raise ValueError(
            f"dict_get: query '{query}' did not match any item in dict:\n{construct_dict_str(dic)}"
        )

    # len(components) == 1
    if components[0] in dic:
        return dic[components[0]]

    if not_exist_ok:
        return default

    raise ValueError(
        f"query '{query}' did not match any item in dict:\n{construct_dict_str(dic)}"
    )


# dict_set sets a value, 'value', which by itself, can be a dict or list or scalar, into 'dic', to become the value of
# the element the path from 'dic' head to which matches 'query' (aka - 'the element specified by the query').
# 'the element specified by the query' is thus either a key in a dictionary element, or a list member specified by
# its index in the list.
# Unless otherwise specified (through 'not_exist_ok=True'), the processing of 'query' by dict_set does not generate
# any new elements into 'dic'. Rather - it just sets the 'value' arg to each and every element the path to which matches
# the query. That 'value' arg, again, can be complex and involved, a dictionary or a list, or scalar, or whatever.
#
# When not_exist_ok = True, dict_set is allowed to generate a new key, or list element, in the element led to
# by the prefix being all but the last component of the query. But not allowed to generate missing earlier components.
#
# If two or more (existing in input dic, or newly generated per not_exist_ok = True) paths in dic match the query (two or
# more paths can match queries that include * components), then all the elements lead to by these paths are assigned
# copies of value.
#
# If set_multiple=True, 'value' must be a list, 'query' should contain at least one * , and there should be exactly
# len(values) paths that match the query, and in this case, dict_set assigns one member of 'value' to each path.
# The matching paths are sorted alphabetically toward the above assignment.
# The processing of set_multiple=True applies to the first occurrence of * in the query, and only to it, and is
# done as follows:
# Let el denote the element lead to by prefix pref of 'query' down to one component preceding the first * in 'query'
# If el is not a list nor a dict, or is a list longer than len('value') or is a dict of len different from len('value'),
# return a failure for prefix pref/*.
# If el is a list shorter than 'value', and not_exist_ok = False, return a failure for prefix pref/*.
# If not_exist_ok = True, and * is the last component in the query, then make el into a list of length
# len('value') that starts with el and continues into zero or more None-s, as many as needed.
# Now that el (potentially wholly or partly generated just now) is a list of length len('value'), set value='value'[i]
# as the target value for the i-th path that goes through el.
# Such a breakdown of 'value' for set_multiple=True, is done only once - on the leftmost * in 'query'.
#
def dict_set(
    dic: dict,
    query: str,
    value: Any,
    not_exist_ok=True,
    set_multiple=False,
):  # sets dic to its new value
    if dic is None or not isinstance(dic, (list, dict)):
        raise ValueError(
            f"Can only change arg dic that is either a dict or a list. Got dic = '{dic}'"
        )

    if query.strip() == "":
        # change the whole input dic, as dic indeed matches ""
        if isinstance(dic, dict):
            if value is None or not isinstance(value, dict):
                raise ValueError(
                    f"Through an empty query, trying to set a whole new value, '{value}', to the whole of dic, '{dic}', but value is not a dict"
                )
            dic.clear()
            dic.update(value)
            return

        if isinstance(dic, list):
            if value is None or not isinstance(value, list):
                raise ValueError(
                    f"Through an empty query, trying to set a whole new value, '{value}', to the whole of dic, '{dic}', but value is not a list"
                )
            dic.clear()
            dic.extend(value)
            return

    if isinstance(dic, dict) and query.strip() in dic:
        dic[query.strip()] = value
        return

    if set_multiple:
        if value is None or not isinstance(value, list) or len(value) == 0:
            raise ValueError(
                f"set_multiple=True, but value, '{value}', can not be broken up, as either it is not a list or it is an empty list"
            )

    components = validate_query_and_break_to_components(query)
    try:
        success, val = set_values(
            current_element=dic,
            value=value,
            index_into_query=(-1) * len(components),
            query=components,
            not_exist_ok=not_exist_ok,
            set_multiple=set_multiple,
        )
        if not success:
            raise ValueError(
                f"dict_set: query '{query}' did not match any item in dict:\n{construct_dict_str(dic)}"
            )

    except Exception as e:
        raise ValueError(
            f"dict_set: query '{query}' did not match any item in dict:\n{construct_dict_str(dic)}"
        ) from e
