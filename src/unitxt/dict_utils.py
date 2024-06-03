import re
from typing import Any, List, Tuple

from .text_utils import construct_dict_str

indx = re.compile(r"^(\d+)$")
name = re.compile(r"^[\w. -]+$")

# formal definition of qpath syntax by which a query is specified:
# qpath -> A (/A)*
# A -> name | * | non-neg-int
# name -> name.matches()
#  * matches ALL members (each and every) of a list or a dictionary element in input dictionary,
#
# a path p in dictionary dic is said to match query qpath if it satisfies the following recursively
# defined condition:
# (1) the prefix of length 0 of p (i.e., pref = "") matches the whole of dic. Also denoted here: pref leads to dic.
# (2) Denoting by el the element in dic lead to by prefix pref of qpath (el must be a list or dictionary),
# and by A (as the definition above) the component, DIFFERENT from *, in qpath, that follows pref, the element
# lead to by pref/A is el[A]. If el[A] is missing from dic, then no path in dic matches prefix pref/A of qpath,
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
# that element according to its arguments, other than not_exist_ok
# If there is not any such element in dic - the function throws or does not throw an exception, depending
# on flag not_exist_ok.
# For a query with *, there could be up to as many as there are values to match the *
# (i.e., length of the list element or dictionary element the children of which match the *; and
# for more than one * in the query -- this effect multiplies)
# Each of the three functions below (dict_get, dict_set, dict_delete) applies the requested
# operation (read, set, or delete) to each and every element el in dic, the path to which matches the query in whole,
# and reads a value from, or sets a new value to, or pops the value out from dic.
#
# If no path in dic matches the query, then # if not_exist_ok=False, the function throws an exception;
# but if not_exist_ok=True, the function returns a default value (dict_get) or does nothing (dict_delete)
# or generates all the needed missing suffixes (dict_set, see details below).
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
        if not (
            bool(name.match(component))
            or component == "*"
            or bool(indx.match(component))
        ):
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
# single component queries without invoking qpath_delete
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
        if component == "*":
            # delete all members of the list or dict
            current_element = [] if isinstance(current_element, list) else {}
            return (True, current_element)
        # component is a either a dictionary key or an index into a list,
        # pop the respective element from current_element
        if indx.match(component):
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
    if indx.match(component):
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
    dic: dict, query: str, not_exist_ok: bool = False, remove_empty_ancestors=False
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
    current_element: Any, query: List[str], index_into_query: int
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
                success, val = get_values(sub_element, query, index_into_query + 1)
                if success:
                    to_ret.append(val)
            except:
                continue

        return (len(to_ret) > 0 or index_into_query == -1, to_ret)
        # when * is the last component, it refers to 'all the contents' of an empty list being current_element.
    # next_component is indx or name, current_element must be a list or a dict
    if indx.match(component):
        component = int(component)
    try:
        success, new_val = get_values(
            current_element[component], query, index_into_query + 1
        )
        if success:
            return (True, new_val)
        return (False, None)
    except:
        return (False, None)


# going down from current_element via query[index_into_query]
# returns the updated current_element
def set_values(
    current_element: Any,
    value: Any,
    index_into_query: int,
    fixed_parameters: dict,
    set_multiple: bool = False,
) -> Tuple[bool, Any]:
    if index_into_query == 0:
        return (True, value)  # matched query all along!

    # current_element should be a list or dict: a containing element
    if current_element is not None and not isinstance(current_element, (list, dict)):
        current_element = None  # give it a chance to become what is needed, if allowed

    if current_element is None and not fixed_parameters["generate_if_not_exists"]:
        return (False, None)

    component = fixed_parameters["query"][index_into_query]
    if component == "*":
        if current_element is not None and set_multiple:
            if isinstance(current_element, dict) and len(current_element) != len(value):
                return (False, None)
            if isinstance(current_element, list) and len(current_element) > len(value):
                return (False, None)
            if len(current_element) < len(value):
                if not fixed_parameters["generate_if_not_exists"]:
                    return (False, None)
                # current_element must be a list, extend current_element to the length needed
                current_element.extend([None] * (len(value) - len(current_element)))
        if current_element is None or current_element == []:
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
                    set_multiple=False,  # now used, not allowed again,
                    fixed_parameters=fixed_parameters,
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
    if indx.match(component):
        if current_element is None or not isinstance(current_element, list):
            if not fixed_parameters["generate_if_not_exists"]:
                return (False, None)
            current_element = []
        # current_element is a list
        component = int(component)
        if component >= len(current_element):
            if not fixed_parameters["generate_if_not_exists"]:
                return (False, None)
            # extend current_element to the length needed
            current_element.extend([None] * (component + 1 - len(current_element)))
        next_current_element = current_element[component]
    else:  # component is a key into a dictionary
        if current_element is None or not isinstance(current_element, dict):
            if not fixed_parameters["generate_if_not_exists"]:
                return (False, None)
            current_element = {}
        if (
            component not in current_element
            and not fixed_parameters["generate_if_not_exists"]
        ):
            return (False, None)
        next_current_element = (
            None if component not in current_element else current_element[component]
        )
    try:
        success, new_val = set_values(
            current_element=next_current_element,
            value=value,
            index_into_query=index_into_query + 1,
            fixed_parameters=fixed_parameters,
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
                f'query "{query}" did not match any item in dict:\n{construct_dict_str(dic)}'
            ) from e

        if not_exist_ok:
            return default

        raise ValueError(
            f'query "{query}" did not match any item in dict:\n{construct_dict_str(dic)}'
        )

    # len(components) == 1
    if components[0] in dic:
        return dic[components[0]]

    if not_exist_ok:
        return default

    raise ValueError(
        f'query "{query}" did not match any item in dict:\n{construct_dict_str(dic)}'
    )


# dict_set sets a value, 'value', which by itself, can be a dict or list or scalar, into 'dic', to become the value of
# the element the path from 'dic' head to which matches 'query'. (aka - 'the element specified by the query')
# 'the element specified by the query' is thus either a key in a dictionary, or a list member specified by its index in the list.
# Unless otherwise specified (through 'not_exist_ok=True'), the processing of 'query' by dict_set does not generate
# any new elements into 'dic'. Rather - it just sets the 'value' arg to each and every element the path to which matches
# the query. That 'value' arg, again, can be complex and involved, a dictionary or a list, or scalar, or whatever.
#
# When not_exist_ok = True, the processing itself is allowed to generate new containing elements (dictionaries, lists, or elements
# therein) into dictionary 'dic', new containing elements such that, at the end of the processing, 'dic' will contain at
# least one element the path to which matches 'query', provided that no existing value in 'dic' is modified nor popped out,
# other than the values sitting on the path along 'query' in whole.
# This generation is defined as follows.
# Having generated what is needed to have in dic an element el, lead to by prefix pref of 'query', and A (as above) is the
# component that follows pref in 'query' (i.e., pref/A is a prefix of 'query', longer than pref by one component) then:
# (1) if indx.match(A), and el existed in 'dic' before dict_set was invoked, then if el is not a list, generate an empty
# list for it: []. If len(el)>A, proceed to element el[A], and continue recursively. If len(el) <= A, extend
# el with [None]*(A+1-len(el)), and continue recursively from there, with elements that surely did not exist in dic
# before dict_set was invoked. If el did not exist in 'dic' before dict_set was invoked, then a whole new list [None]*(A+1)
# is generated, and continue from there recursively.
# (2) if not indx but name.match(A), continue in analogy with (1), with el being a dictionary now.
# (3) if A is '*', and el already exists, continue into ALL el's existing sub_elements as above. if el was not existing
# in dic before dict_set was invoked, which means it is None that we ride on from (1) or (2), then we generate
# a new [None] List (only a list, not a dict, because we have no keys to offer), and continue as above
# once the end of the query is thus reached, 'value' is returned backward on the recursion, and the elements
# that were None for a while - reshape into the needed (dict or list) element.
#
# If two or more (existing in input dic, or newly generated per not_exist_ok = True) paths in dic match the query
# all the elements lead to by these paths are assigned copies of value.
#
# If set_multiple=True, 'value' must be a list, 'query' should contain at least one * , and there should be exactly
# len(values) paths that match the query, and in this case, dict_set assigns one member of 'value' to each path.
# The matching paths are sorted alphabetically toward the above assignment.
# The processing of set_multiple=True applies to the first occurrence of * in the query, and only to it, and is
# done as follows:
# Let el denote the element lead to by prefix pref of 'query' down to one component preceding the first * in 'query'
# (depending on not_exist_ok, el can be None). If el existed in dic before dict_set is invoked (el is not None) and
# is not a list nor a dict, or is a list longer than len('value') or is a dict of len different from len('value'),
# return a failure for prefix pref/*.
# If el existed, and is a list shorted than len('value'), or did not exist at all, then if not_exist_ok= False,
# return a failure for prefix pref/*. If not_exist_ok = True, then make el into a list of length
# len('value') that starts with el (if existed) and continues into zero or more None-s, as many as needed.
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
            f"Can not change dic that is either None or not a dict nor a list. Got dic = {dic}"
        )

    if query.strip() == "":
        # change the whole input dic, as dic indeed matches ""
        if isinstance(dic, dict):
            if value is None or not isinstance(value, dict):
                raise ValueError(
                    f"Through an empty query, trying to set a whole new value, {value}, to the whole of dic, {dic}, but value is not a dict"
                )
            dic.clear()
            dic.update(value)
            return

        if isinstance(dic, list):
            if value is None or not isinstance(value, list):
                raise ValueError(
                    f"Through an empty query, trying to set a whole new value, {value}, to the whole of dic, {dic}, but value is not a list"
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
                f"set_multiple=True, but value, {value}, can not be broken up, as either it is not a list or it is an empty list"
            )

    components = validate_query_and_break_to_components(query)
    fixed_parameters = {
        "query": components,
        "generate_if_not_exists": not_exist_ok,
    }
    try:
        success, val = set_values(
            current_element=dic,
            value=value,
            index_into_query=(-1) * len(components),
            fixed_parameters=fixed_parameters,
            set_multiple=set_multiple,
        )
        if not success and not not_exist_ok:
            raise ValueError(f"No path in dic {dic} matches query {query}.")

    except Exception as e:
        raise ValueError(f"No path in dic {dic} matches query {query}.") from e
