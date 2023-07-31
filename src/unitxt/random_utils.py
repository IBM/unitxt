import contextlib
import random as python_random
import string
import threading

__default_seed__ = 42
_thread_local = threading.local()
_thread_local.seed = __default_seed__
_thread_local.random = python_random.Random()
random = _thread_local.random


def set_seed(seed):
    _thread_local.random.seed(seed)
    _thread_local.seed = seed


def get_seed():
    return _thread_local.seed


def get_random_string(length):
    letters = string.ascii_letters
    result_str = "".join(random.choice(letters) for _ in range(length))
    return result_str


@contextlib.contextmanager
def nested_seed(sub_seed=None):
    state = _thread_local.random.getstate()
    old_global_seed = get_seed()
    sub_seed = sub_seed or get_random_string(10)
    new_global_seed = str(old_global_seed) + "/" + sub_seed
    set_seed(new_global_seed)
    try:
        yield _thread_local.random
    finally:
        set_seed(old_global_seed)
        _thread_local.random.setstate(state)
