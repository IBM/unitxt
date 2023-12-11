import os
from typing import Sequence

import dpath
from dpath import MutableSequence
from dpath.segments import extend


def is_subpath(subpath, fullpath):
    # Normalize the paths to handle different formats and separators
    subpath = os.path.normpath(subpath)
    fullpath = os.path.normpath(fullpath)

    # Split the paths into individual components
    subpath_components = subpath.split(os.path.sep)
    fullpath_components = fullpath.split(os.path.sep)

    # Check if the full path starts with the subpath
    return fullpath_components[: len(subpath_components)] == subpath_components


def dpath_get(dic, query_path):
    return [v for _, v in dpath.search(dic, query_path, yielded=True)]


def dpath_set_one(dic, query_path, value):
    n = dpath.set(dic, query_path, value)
    if n != 1:
        raise ValueError(f'query "{query_path}" matched multiple items in dict: {dic}')


def dict_delete(dic, query_path):
    dpath.delete(dic, query_path)


def dict_creator(current, segments, i, hints=()):
    """Create missing path components. If the segment is an int, then it will create a list. Otherwise a dictionary is created.

    set(obj, segments, value) -> obj
    """
    segment = segments[i]
    length = len(segments)

    if isinstance(current, Sequence):
        segment = int(segment)

    if isinstance(current, MutableSequence):
        extend(current, segment)

    # Infer the type from the hints provided.
    if i < len(hints):
        current[segment] = hints[i][1]()
    else:
        # Peek at the next segment to determine if we should be
        # creating an array for it to access or dictionary.
        if i + 1 < length:
            segment_next = segments[i + 1]
        else:
            segment_next = None

        if isinstance(segment_next, int) or (
            isinstance(segment_next, str) and segment_next.isdecimal()
        ):
            current[segment] = []
        else:
            current[segment] = {}


def dpath_set(dic, query_path, value, not_exist_ok=True):
    paths = [p for p, _ in dpath.search(dic, query_path, yielded=True)]
    if len(paths) == 0 and not_exist_ok:
        dpath.new(dic, query_path, value, creator=dict_creator)
    else:
        if len(paths) != 1:
            raise ValueError(
                f'query "{query_path}" matched {len(paths)} items in dict: {dic}. should match only one.'
            )
        for path in paths:
            dpath_set_one(dic, path, value)


def dpath_set_multiple(dic, query_path, values, not_exist_ok=True):
    paths = [p for p, _ in dpath.search(dic, query_path, yielded=True)]
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
    for path, value in zip(paths, values):
        dpath_set_one(dic, path, value)


def dict_get(dic, query, use_dpath=True, not_exist_ok=False, default=None):
    if use_dpath:
        values = dpath_get(dic, query)
        if len(values) == 0 and not_exist_ok:
            return default
        if len(values) == 0:
            raise ValueError(f'query "{query}" did not match any item in dict: {dic}')

        if len(values) == 1 and "*" not in query and "," not in query:
            return values[0]

        return values

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
            dpath_set(dic, query, value, not_exist_ok=not_exist_ok)
    else:
        dic[query] = value
