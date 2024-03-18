import hashlib
import random as python_random
from typing import Any, Hashable

from .settings_utils import get_settings

settings = get_settings()


def get_seed():
    return settings.seed


def new_random_generator(sub_seed: Any) -> python_random.Random:
    """Get a generator based on a seed derived from the default seed.

    The purpose is to have a random generator that provides outputs
    that are independent of previous randomizations.
    """
    if not isinstance(sub_seed, Hashable):
        # e.g. for lists or dicts
        # Create a persistent hash for the input object (using plain hash(..) produces
        # a value that varies between runs)
        sub_seed_str = str(sub_seed).encode("utf-8")
        # limit the hash int size to 2^32
        sub_seed_hexdigest = hashlib.md5(sub_seed_str).hexdigest()[:8]
        # convert to int, from base 16:
        sub_seed_int = int(sub_seed_hexdigest, 16)
        sub_seed = str(sub_seed_int)
    elif not isinstance(sub_seed, str):
        # for Hashable objects that are not strings
        sub_seed = str(hash(sub_seed))

    sub_default_seed = str(settings.seed) + "/" + sub_seed
    return python_random.Random(sub_default_seed)
