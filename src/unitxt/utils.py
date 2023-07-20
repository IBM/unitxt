from typing import Dict, Any
from queue import Queue


def is_done(query):
    return query is None or len(query) == 0 or query == '/'


def is_wildcard(query):
    return query == '*'


def is_int(query):
    return query.isdigit()

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_query(dic, query_path):
    results = []
    tasks = Queue()

    tasks.put((dic, query_path, ''))

    while not tasks.empty():

        d, q, p = tasks.get()

        if is_done(q):
            results.append(d)
        else:

            parts = q.split('/', maxsplit=1)

            current = parts[0]
            nexts = parts[1] if len(parts) > 1 else None

            if is_wildcard(current):
                assert isinstance(d, list), f'cannot query wildcard "*" on non-list object: {d} (in path: {p})'
                for item in d:
                    tasks.put((item, nexts, p + f'/{item}'))

            elif is_int(current):
                assert isinstance(d, list), f'cannot query integer "{current}" on non-list object: {d} (in path: {p})'
                tasks.put((d[int(current)], nexts, p + f'/{current}'))

            elif not isinstance(d, dict):
                raise ValueError(f'cannot query "{current}" on non-dict object: {d} (in path: {p})')
            else:
                if current not in d:
                    raise ValueError(f'"{current}" not in dict object: {d} (in path: {p})')
                tasks.put((d[current], nexts, p + f'/{current}'))

    if len(results) == 0:
        raise ValueError(f'query "{query_path}" did not match any item in dict: {dic}')

    return results if len(results) > 1 else results[0]
