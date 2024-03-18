import re
from typing import Any, List, Tuple

# import dpath
# from dpath import MutableSequence
# from dpath.segments import extend

indx = re.compile(r"^\[?(\d+)\]?$")
name = re.compile(r"^(\w+)$")

# formal definition of qpath syntax:
# qpath -> A (/A)*
# A -> name | * | [?non-neg-int]?
# name -> alphanumeric


def validate_qpath(qpath: str) -> bool:
    qpath = qpath.replace(" ", "")
    if qpath.endswith("/") or "//" in qpath or qpath.startswith("["):
        raise ValueError(
            f"qpath should not contain // nor start with [, nor end with /. Got {qpath}"
        )
    if qpath.startswith("/"):
        qpath = qpath[1:]
    components = qpath.split("/")
    components = [component.strip() for component in components]
    for component in components:
        if not (
            bool(name.match(component))
            or component == "*"
            or bool(indx.match(component))
        ):
            return False
    return True


def is_subpath(subpath, fullpath):
    # Split the paths into individual components
    subpath_components = subpath.split("/")
    fullpath_components = fullpath.split("/")
    subpath_components = [component.strip() for component in subpath_components]
    fullpath_components = [component.strip() for component in fullpath_components]

    # Check if the full path starts with the subpath
    return fullpath_components[: len(subpath_components)] == subpath_components


def consume_component(qpath: str) -> Tuple[str, str]:
    assert (
        len(qpath.strip()) > 0
    ), "should not try to consume a component from an empty qpath"
    if "/" in qpath:
        component = qpath[: qpath.find("/")]
        qpath = qpath[1 + qpath.find("/") :]
        return (component.strip(), qpath.strip())
    return (qpath.strip(), "")  # nothing left


# frontier is a list of pairs. Each pair consists of a string path (that is a subpath of qpath),
# and a value that is the resident of dic at the position lead to by the string path.
# flake8: noqa: C901


def verify_for_list(obj: Any, comp: str) -> int:
    if not isinstance(obj, list):
        raise ValueError(f"qpath specifies an index {comp} for {obj!s}")
    i = int(re.match(indx, comp).group(1))
    if i >= len(obj):
        raise ValueError(f"qpath specifies an index {comp} for a shorter list: {obj!s}")
    return i


def advance_frontier(
    frontier: List[Tuple[str, Any]], comp: str
) -> List[Tuple[str, Any]]:
    new_frontier = []
    for f in frontier:
        if re.match(indx, comp):
            i = verify_for_list(f[1], comp)
            # if not isinstance(f[1], list):
            #     raise ValueError(f"qpath specifies an index {comp} for {f[1]!s}")
            # i = int(re.match(indx, comp).group(1))
            # if i >= len(f[1]):
            #     raise ValueError(
            #         f"qpath specifies an index {comp} for a shorter list: {f[1]!s}"
            #     )
            new_frontier.append((f[0] + f"/[{i}]", f[1][i]))
            continue
        if bool(name.match(comp)) and comp in f[1]:
            new_frontier.append(
                (f[0] + ("/" if len(f[0]) > 0 else "") + comp, f[1][comp])
            )
            continue
        if comp == "*":
            if not isinstance(f[1], (list, dict)):
                raise ValueError(f"qpath specifies * for {f[1]!s}")
            if isinstance(f[1], dict):
                for key in f[1]:
                    new_frontier.append(
                        (f[0] + ("/" if len(f[0]) > 0 else "") + key, f[1][key])
                    )
                continue
            # f[1] is a list
            for i in range(len(f[1])):
                new_frontier.append((f[0] + f"/[{i}]", f[1][i]))
            continue
        return []  # no path found
    return new_frontier


# return all the values sitting inside dic, in all the paths that match qpath,
# paths being sorted lexicographically
# return them as pairs of: (path, value). None of the paths contains *: these are all
# elaborated and listed out


def qpath_get(dic: dict, query_path: str) -> List[Tuple[str, Any]]:
    if not validate_qpath(query_path):
        raise ValueError(f"Received an invalid qpath: {query_path}")
    if query_path.startswith("/"):
        query_path = query_path[1:]
    frontier = [("", dic)]
    qp = query_path
    while len(qp) > 0:
        comp, qp = consume_component(qp)
        frontier = advance_frontier(frontier, comp)

    # now sort frontier by lexicographical order of its paths
    frontier.sort(key=lambda front: front[0])
    return frontier


# def dpath_get(dic, query_path):
#     return [v for _, v in dpath.search(dic, query_path, yielded=True)]


def dpath_set_one(dic, query_path, value):
    qpath_set(dic=dic, qpath=query_path, value=value)
    # n = dpath.set(dic, query_path, value)
    # if n != 1:
    #     raise ValueError(f'query "{query_path}" matched multiple items in dict: {dic}')


# qpath is fully specified, no *-s.
def get_parent(
    dic: dict, qpath: str
) -> Tuple[
    str, str, Any
]:  # child_name, path to parent, element (list or dict) being the parent
    if not validate_qpath(qpath) or "*" in qpath:
        raise ValueError(
            f"Received an invalid qpath, or a path that contains *, '{qpath}', and hence might specify more than one child. Parent can not be determined."
        )
    if qpath.startswith("/"):
        qpath = qpath[1:]
    if "/" not in qpath:
        return (qpath, "", dic)
    parent_path = qpath[: qpath.rfind("/")]
    child_name = qpath[1 + qpath.rfind("/") :]
    parents = qpath_get(dic, parent_path)
    if len(parents) != 1:
        raise ValueError(f"More than one parent found to {qpath}. Check this error")
    return (child_name, parents[0][0], parents[0][1])


def dict_delete(dic: dict, qpath: str):
    # we remove from dic all the elements lead to by the path.
    # If the removal of any such element leaves its parent (list or dict) within dic empty -- remove that parent as well.

    if "/" not in qpath:
        if qpath not in dic:
            raise ValueError(
                f"And attempt to delete from dictionary {dic}, an element {qpath}, that does not exist in the dictionary"
            )
        dic.pop(qpath)
        return
    to_deletes = qpath_get(dic, qpath)
    for to_delete in to_deletes:
        dict_delete_one_element(dic, to_delete[0])


# here qpath does not contain any *. it leads to only one element in dic, potentially, a list element.
# If the removal of the element element leaves its parent (list or dict) within dic empty -- remove that parent as well.
#
def dict_delete_one_element(dic: dict, qpath: str):
    if not validate_qpath(qpath) or "*" in qpath:
        raise ValueError(
            "invalid query, or a query that might lead to more than one element"
        )
    if qpath.startswith("/"):
        qpath = qpath[1:]
    checking = qpath_get(dic, qpath)
    if len(checking) != 1:
        raise ValueError(
            f"qpath '{qpath}' does not specify a single path in dictionary '{dic}'. It specifies {len(checking)} paths."
        )
    qp = qpath
    while True:
        child_name, path_to_parent, parent = get_parent(dic, qp)
        if bool(name.match(child_name)):
            parent.pop(child_name)
        else:  # parent is a list, and child name is an index:
            i = int(child_name[1:-1])
            if not isinstance(parent, list) or i >= len(parent):
                raise ValueError(
                    f"An attempt to remove the {i}-th element from a shorter list {parent}"
                )
            parent.pop(i)
        if parent or len(path_to_parent) == 0:
            break
        # need to delete parent too
        qp = path_to_parent


# def dict_creator(current, segments, i, hints=()):
#     """Create missing path components. If the segment is an int, then it will create a list. Otherwise a dictionary is created.
#
#     set(obj, segments, value) -> obj
#     """
#     segment = segments[i]
#     length = len(segments)
#
#     if isinstance(current, Sequence):
#         segment = int(segment)
#
#     if isinstance(current, MutableSequence):
#         extend(current, segment)
#
#     # Infer the type from the hints provided.
#     if i < len(hints):
#         current[segment] = hints[i][1]()
#     else:
#         # Peek at the next segment to determine if we should be
#         # creating an array for it to access or dictionary.
#         if i + 1 < length:
#             segment_next = segments[i + 1]
#         else:
#             segment_next = None
#
#         if isinstance(segment_next, int) or (
#             isinstance(segment_next, str) and segment_next.isdecimal()
#         ):
#             current[segment] = []
#         else:
#             current[segment] = {}


def build_bottom_up(components: List[str], value: Any) -> Any:
    pointer = value
    for i in range(len(components) - 1, -1, -1):
        component = components[i]
        if bool(name.match(component)):
            pointer = {component: pointer}
        else:  # an index,  * is not allowed here
            j = int(component[1:-1])
            if j > 0:
                raise ValueError(
                    "trying to generate a new list node, with a given element to be assigned, but the position given to it is not 0."
                )
            pointer = [pointer]
    return pointer


# sets a single values (that each by itself could be a list or dict) into dic,
# at the position lead to by qpath.
# qpath should lead to a single position (or multiple, if the last one
# if these positions are not yet born, and not_exist_ok == True: these positions are enerated inside dic
# flake8: noqa: C901
def qpath_set(dic: dict, qpath: str, value: Any, not_exist_ok: bool = True):
    if not validate_qpath(qpath) or "*" in qpath:
        raise ValueError(
            f"Invalid query, {qpath}, or a query that might lead to more than one position in dic"
        )
    if qpath.startswith("/"):
        qpath = qpath[1:]
    components = qpath.split("/")
    components = [component.strip() for component in components]
    pointer = dic
    for i, component in enumerate(components):
        if bool(name.match(component)):
            if not isinstance(pointer, dict):
                raise ValueError(
                    f"path {qpath} leads down into {component}, but from a position that is not a dictionary"
                )
            if component in pointer:
                if i == len(components) - 1:
                    pointer[component] = value
                    return
                pointer = pointer[component]
            else:
                if not_exist_ok:
                    pointer[component] = build_bottom_up(components[i + 1 :], value)
                    return
                raise ValueError(
                    f"not_exist_ok == False, but can not get down to component {component}, in dic: {dic}, along qpath {qpath}"
                )
        else:  # an index, * is not allowed here
            if not isinstance(pointer, list):
                raise ValueError(
                    f"path {qpath} leads down into position {component}, but from a position that is not a list"
                )
            j = int(component[1:-1])
            if j < len(pointer):
                if i == len(components) - 1:
                    pointer[j] = value
                    return
                pointer = pointer[j]
            else:
                if not_exist_ok and j == len(pointer):
                    pointer.append(build_bottom_up(components[i + 1 :], value))
                    return
                raise ValueError(
                    f"not_exist_ok == False, but can not get down to index {component}, in dic: {dic}, along qpath {qpath}"
                )


# def dpath_set(dic, query_path, value, not_exist_ok=True):
#     paths = [p for p, _ in dpath.search(dic, query_path, yielded=True)]
#     if len(paths) == 0 and not_exist_ok:
#         dpath.new(dic, query_path, value, creator=dict_creator)
#     else:
#         if len(paths) != 1:
#             raise ValueError(
#                 f'query "{query_path}" matched {len(paths)} items in dict: {dic}. should match only one.'
#             )
#         for path in paths:
#             dpath_set_one(dic, path, value)
#
#
# def qpath_set_multiple(dic, query_path, values, not_exist_ok=True):
#     pass


def dpath_set_multiple(dic, query_path, values, not_exist_ok=True):
    # paths = [p for p, _ in dpath.search(dic, query_path, yielded=True)]
    paths = qpath_get(dic, query_path)
    if len(paths) == 0:
        if not_exist_ok:
            raise ValueError(
                f"Cannot set multiple values to non-existing path: {query_path}"
            )
        raise ValueError(f'query "{query_path}" did not match any item in dict: {dic}')

    if len(paths) != len(values):
        raise ValueError(
            f'query "{query_path}" matched {len(paths)} items in dict: {dic} but {len(values)} values are provided. should match only one.'
        )
    paths = [path[0] for path in paths]  # from list of frontiers to a list of paths
    for path, value in zip(paths, values):
        dpath_set_one(dic, path, value)


# the returned values are ordered by the lexicographic order of the paths leading to them
def dict_get(dic, query, use_dpath=True, not_exist_ok=False, default=None):
    if use_dpath:
        values = qpath_get(dic, query)  # a list of tuples (path to value, value)
        if len(values) == 0 and not_exist_ok:
            return default
        if len(values) == 0:
            raise ValueError(f'query "{query}" did not match any item in dict: {dic}')

        to_ret = [v[1] for v in values]
        if len(to_ret) == 1:
            return to_ret[0]
        return to_ret

    if not_exist_ok:
        return dic.get(query, default)

    if query in dic:
        return dic[query]

    raise ValueError(f'query "{query}" did not match any item in dict: {dic}')


def dict_set(dic, query, value, use_dpath=True, not_exist_ok=True, set_multiple=False):
    if use_dpath:
        if set_multiple:
            dpath_set_multiple(dic, query, value, not_exist_ok=not_exist_ok)
        else:
            qpath_set(dic, query, value, not_exist_ok=not_exist_ok)
    else:
        dic[query] = value
