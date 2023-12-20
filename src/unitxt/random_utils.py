import random as python_random

__default_seed__ = 42

from typing import Any, Hashable


def get_seed():
    return __default_seed__


def new_random_generator(sub_seed: Any) -> python_random.Random:
    """Get a generator based on a seed derived from the default seed.

    The purpose is to have a random generator that provides outputs
    that are independent of previous randomizations.
    """
    if not isinstance(sub_seed, Hashable):
        # e.g. for lists or dicts
        sub_seed = str(hash(str(sub_seed)))
    elif not isinstance(sub_seed, str):
        # for Hashable objects that are not strings
        sub_seed = str(hash(sub_seed))

    sub_default_seed = str(__default_seed__) + "/" + sub_seed
    return python_random.Random(sub_default_seed)
