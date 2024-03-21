import re
from typing import Any, List

indx = re.compile(r"^(\d+)$")
name = re.compile(r"^[\w. -]+$")

# formal definition of qpath syntax:
# qpath -> A (/A)*
# A -> name | * | non-neg-int
# name -> name.matches()
# * should only replace indexes of a list, not all fields of a dictionary


# validate and normalize
# leading / is ignored, all paths are treated as coming from root of dic
# trailing / is ignored
def validate_query_and_break_to_components(query: str) -> List[str]:
    query = query.replace("//", "/").strip()
    if query.startswith("/"):
        query = query[1:]  # remove the leading /

    if query.endswith("/"):
        query = query + "*"
        # same meaning, and make sure the / not lost when splitting

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


def qpath_delete(
    current_element: dict,
    query: List[str],
    index_into_query: int,
    remove_empty_ancestors=False,
):
    component = query[index_into_query]
    if index_into_query == -1:
        if component == "*":
            current_element = []
        else:
            # component is a name or an index
            if indx.match(component):
                component = int(component)
            current_element.pop(component)  # thrown exception will be caught outside
        return (
            None
            if remove_empty_ancestors and len(current_element) == 0
            else current_element
        )

    # index_into_query < -1
    if component == "*":
        current_element = [
            qpath_delete(
                current_element=sub_element,
                query=query,
                index_into_query=index_into_query + 1,
                remove_empty_ancestors=remove_empty_ancestors,
            )
            for sub_element in current_element
        ]
        current_element = [elem for elem in current_element if elem is not None]
    elif indx.match(component):  # thrown exception will be caught
        current_element[int(component)] = qpath_delete(
            current_element=current_element[int(component)],
            query=query,
            index_into_query=index_into_query + 1,
            remove_empty_ancestors=remove_empty_ancestors,
        )
        if current_element[int(component)] is None:
            current_element.pop(int(component))
    elif name.match(component):  # thrown exception will be caught
        current_element[component] = qpath_delete(
            current_element=current_element[component],
            query=query,
            index_into_query=index_into_query + 1,
            remove_empty_ancestors=remove_empty_ancestors,
        )
        if current_element[component] is None:
            current_element.pop(component)
    return (
        None
        if remove_empty_ancestors and len(current_element) == 0
        else current_element
    )


def dict_delete(dic: dict, query: str, remove_empty_ancestors=False):
    # we remove from dic all the elements lead to by the path.
    # If remove_empty_ancestors=True, and the removal of any such element leaves its parent (list or dict)
    # within dic empty -- remove that parent as well, and continue recursively
    # Treating trailing /* :
    # delete  references/* means just to shrink to [] the list of children of references
    # delete  references/ means the same as delete references/*
    # delete references, pops references out from dic
    # the above -- before dealing with removal of empty ancestors.
    if "/" not in query:
        if query.strip() not in dic:
            raise ValueError(
                f"An attempt to delete from dictionary {dic}, an element {query}, that does not exist in the dictionary"
            )
        dic.pop(query.strip())
        return
    qpath = validate_query_and_break_to_components(query)
    try:
        qpath_delete(
            current_element=dic,
            query=qpath,
            index_into_query=(-1) * len(qpath),
            remove_empty_ancestors=remove_empty_ancestors,
        )
    except Exception as e:
        raise ValueError(f"query {query} matches no path in dictionary {dic}") from e


# returns all the values sitting inside dic, in all the paths that match query_path
# if query includes * then return a list of values reached by all paths that match the query
# flake8: noqa: C901
def get_values(
    current_element: Any, query: List[str], index_into_query: int
) -> List[Any]:
    # going down from current_element through query[index_into_query].
    if index_into_query == 0:
        return current_element
    component = query[index_into_query]
    # index_into_query < 0
    if component == "*":
        # current_element must be a list or a dictionary, otherwise, an exception will be thrown, and cuaght outside
        if not isinstance(current_element, (list, dict)):
            raise ValueError(
                f"* in query {query}, when evaluated over the input dictionary, does not correspond to a dict nor a list"
            )
        to_ret = []
        for sub_element in current_element:
            if isinstance(current_element, dict):
                to_ret.append(
                    get_values(
                        current_element[sub_element], query, index_into_query + 1
                    )
                )
            else:
                to_ret.append(get_values(sub_element, query, index_into_query + 1))
        return to_ret
    # next_component is indx or name
    if indx.match(component):
        component = int(component)
        if component >= len(current_element):
            raise ValueError(
                f"Trying to fetch element in position {component} from a shorter list: {current_element}"
            )
        return get_values(current_element[component], query, index_into_query + 1)
    # next_component is a name
    if component not in current_element:
        raise ValueError(
            f"Trying to look up key {component} in a dictionary {current_element} that misses that key"
        )
    return get_values(current_element[component], query, index_into_query + 1)


# going down from current_element via query[index_into_query]
# returns the updated current_element
def set_values(
    current_element: Any,
    value: Any,
    index_into_query: int,
    fixed_parameters: dict,
    set_individual_already_used: bool = False,
):
    assert (
        current_element or fixed_parameters["generate_if_not_exists"]
    ), "Current_element == None, and yet not generate_if_not_exists. Check who sent us here"
    component = fixed_parameters["query"][index_into_query]
    if component == "*":
        # it is not a trailing * of the original query, that one, if there was any, was removed on starting of processing
        if set_individual_already_used:
            # set_individual_already_used indicates that generate_if_not_exists == True and current_element == None
            assert (
                not current_element and fixed_parameters["generate_if_not_exists"]
            ), "Should not happen. Dafna to check"
            raise ValueError(
                "Can not respect * more than once along the path for generation of new list of length determined by the length of the input value"
            )
        if not current_element and not fixed_parameters["set_individual"]:
            raise ValueError(
                "Can not tell the length of the list to generate for a * that does not end the input query, when set_multiple == False"
            )

    if component == "*":
        if current_element:
            if fixed_parameters["set_individual"] and len(current_element) != len(
                value
            ):
                raise ValueError(
                    "set_multiple == True, while an existing element matching a * has a different len than values. dic: {dic}, query: {query}, value: {value}"
                )
            for i in range(len(current_element)):
                current_element[i] = (
                    value
                    if index_into_query == -1
                    else set_values(
                        current_element=current_element[i],
                        value=value[i] if fixed_parameters["set_individual"] else value,
                        index_into_query=index_into_query + 1,
                        fixed_parameters=fixed_parameters,
                        set_individual_already_used=set_individual_already_used,
                    )
                )
            return current_element
        a_list = [None] * len(value)
        # not current_element, and we are now using set_individual:
        for i in range(len(a_list)):
            a_list[i] = (
                (value if not fixed_parameters["set_individual"] else value[i])
                if index_into_query == -1
                else set_values(
                    current_element=None,
                    value=value[i],
                    index_into_query=index_into_query + 1,
                    fixed_parameters=fixed_parameters,
                    set_individual_already_used=True,
                )
            )
        return a_list
    if indx.match(component):
        component = int(component)
        if current_element and component < len(current_element):
            current_element[component] = (
                value
                if index_into_query == -1
                else set_values(
                    current_element=current_element[component]
                    if isinstance(current_element[component], (dict, list))
                    else None,
                    value=value,
                    index_into_query=index_into_query + 1,
                    fixed_parameters=fixed_parameters,
                    set_individual_already_used=set_individual_already_used,
                )
            )
            return current_element
        if not fixed_parameters["generate_if_not_exists"]:
            raise ValueError(
                "Trying to set a value into dic {dic} via a query {query} that matches no path, while not_exist_ok=False"
            )
        a_list = [None] * (
            (1 + component)
            if not current_element
            else (1 + component - len(current_element))
        )
        a_list[component - (0 if not current_element else len(current_element))] = (
            value
            if index_into_query == -1
            else set_values(
                current_element=None,
                value=value,
                index_into_query=index_into_query + 1,
                fixed_parameters=fixed_parameters,
                set_individual_already_used=set_individual_already_used,
            )
        )

        if not current_element:
            return a_list  # the updated value of current_element:
        current_element.extend(a_list)
        return current_element

    # query[index_into_query] must be a name
    if (
        not current_element or component not in current_element
    ) and not fixed_parameters["generate_if_not_exists"]:
        raise ValueError(
            "Trying to set a value into dic {dic} via a query {query} that matches no path, while not_exist_ok=False"
        )

    a_dict = current_element if current_element else {}
    a_dict[component] = (
        value
        if index_into_query == -1
        else set_values(
            current_element=a_dict[component]
            if component in a_dict and isinstance(a_dict[component], (dict, list))
            else None,
            value=value,
            index_into_query=index_into_query + 1,
            fixed_parameters=fixed_parameters,
            set_individual_already_used=set_individual_already_used,
        )
    )
    return a_dict


# the returned values are ordered by the lexicographic order of the paths leading to them
def dict_get(
    dic: dict,
    query: str,
    use_dpath: bool = True,
    not_exist_ok: bool = False,
    default: Any = None,
):
    if use_dpath and "/" in query:
        components = validate_query_and_break_to_components(query)
        if len(components) == 1:
            if components[0] in dic:
                return dic[components[0]]
            if not_exist_ok:
                return default
            raise ValueError(f'query "{query}" did not match any item in dict: {dic}')
        try:
            values = get_values(dic, components, -1 * len(components))
        except Exception as e:
            if not_exist_ok:
                return default
            raise ValueError(
                f'query "{query}" did not match any item in dict: {dic}'
            ) from e

        if isinstance(values, list) and len(values) == 0:
            if not_exist_ok:
                return default
            raise ValueError(f'query "{query}" did not match any item in dict: {dic}')

        return values

    if query.strip() in dic:
        return dic[query.strip()]

    if not_exist_ok:
        return default

    raise ValueError(f'query "{query}" did not match any item in dict: {dic}')


# dict_set sets a value, which by itself, can be a dict or list or scalar, into dic, in the element specified by query.
# if not_exist_ok = True, and such a path does not exist in dic, this function also builds that path inside dic, and then
# assigns the value.
# if two or more (existing in input dic, or newly generated per not_exist_ok = True) paths in dic match the query
# all these paths are assigned copies of value.
# if set_multiple=True, value must be a list, and there should be exactly len(values) paths matching of query,
# and in this case, function assigns one member of list value to each path.
# the matching paths are sorted alphabetically toward the above assignment
def dict_set(
    dic: dict,
    query: str,
    value: Any,
    use_dpath=True,
    not_exist_ok=True,
    set_multiple=False,
):
    if (set_multiple or query.strip().endswith("*")) and (not isinstance(value, list)):
        raise ValueError(
            f"set_multiple == True, or a 'whole list' is expected to be assigned, and yet value, {value}, is not a list"
        )
    if use_dpath and "/" in query:
        components = validate_query_and_break_to_components(query)
        if components[-1] == "*":
            # value is a list, and it is to be assigned to the parent of this * in the query
            components = components[:-1]
        fixed_parameters = {
            "query": components,
            "generate_if_not_exists": not_exist_ok,
            "set_individual": set_multiple,
        }
        try:
            dic = set_values(
                current_element=dic,
                value=value,
                index_into_query=(-1) * len(components),
                fixed_parameters=fixed_parameters,
                set_individual_already_used=False,
            )
        except Exception as e:
            raise ValueError(str(e).format(query=query, dic=dic, value=value)) from e
    else:
        if query not in dic and not not_exist_ok:
            raise ValueError(
                f"not_exist_ok=False and query {query} matches no path in dic {dic}, still was trying to set a value into dic by the query"
            )
        dic[query] = value
