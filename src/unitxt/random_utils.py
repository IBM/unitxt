import random as python_random
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


def get_sub_default_random_generator(sub_seed: str) -> python_random.Random:
    """Get a generator based on a seed derived from the default seed.

    The purpose is to have a random generator that provides outputs
    that are independent of previous randomizations.
    """
    sub_default_seed = str(__default_seed__) + "/" + sub_seed
    return python_random.Random(sub_default_seed)
