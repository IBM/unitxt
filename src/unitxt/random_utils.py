import random as python_random

__default_seed__ = 42


def get_seed():
    return __default_seed__


def new_random_generator(sub_seed: str) -> python_random.Random:
    """Get a generator based on a seed derived from the default seed.

    The purpose is to have a random generator that provides outputs
    that are independent of previous randomizations.
    """
    sub_default_seed = str(__default_seed__) + "/" + sub_seed
    return python_random.Random(sub_default_seed)
