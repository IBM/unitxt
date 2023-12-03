import contextlib
import random as python_random
import string
import threading

__default_seed__ = 42
_thread_local = threading.local()


def get_seed():
    try:
        return _thread_local.seed
    except AttributeError:
        _thread_local.seed = __default_seed__
        return _thread_local.seed


def get_random():
    try:
        return _thread_local.random
    except AttributeError:
        _thread_local.random = python_random.Random(get_seed())
        return _thread_local.random


random = get_random()


def set_seed(seed):
    _thread_local.seed = seed
    get_random().seed(seed)


def get_random_string(length):
    letters = string.ascii_letters
    return "".join(get_random().choice(letters) for _ in range(length))


@contextlib.contextmanager
def nested_seed(sub_seed=None):
    old_state = get_random().getstate()
    old_global_seed = get_seed()
    sub_seed = sub_seed or get_random_string(10)
    new_global_seed = str(old_global_seed) + "/" + sub_seed
    set_seed(new_global_seed)
    try:
        yield get_random()
    finally:
        set_seed(old_global_seed)
        get_random().setstate(old_state)
