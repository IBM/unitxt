import copy
import functools
import importlib.util
import json
import os
import random
import re
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict
from urllib.error import HTTPError as UrllibHTTPError

from requests.exceptions import ConnectionError, HTTPError
from requests.exceptions import Timeout as TimeoutError

from .logging_utils import get_logger
from .settings_utils import get_settings
from .text_utils import is_made_of_sub_strings

logger = get_logger()
settings = get_settings()


def retry_connection_with_exponential_backoff(
    max_retries=None,
    retry_exceptions=(
        ConnectionError,
        TimeoutError,
        HTTPError,
        FileNotFoundError,
        UrllibHTTPError,
    ),
    backoff_factor=1,
):
    """Decorator that implements retry with exponential backoff for network operations.

    Also handles errors that were triggered by the specified retry exceptions,
    whether they're direct causes or part of the exception context.

    Args:
        max_retries: Maximum number of retry attempts (falls back to settings if None)
        retry_exceptions: Tuple of exceptions that should trigger a retry
        backoff_factor: Base delay factor in seconds for backoff calculation

    Returns:
        The decorated function with retry logic
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get max_retries from settings if not provided
            retries = (
                max_retries
                if max_retries is not None
                else settings.max_connection_retries
            )

            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if this exception or any of its causes match the retry exceptions
                    should_retry = False
                    current_exc = e

                    # Check the exception chain for both __cause__ (explicit) and __context__ (implicit)
                    visited_exceptions = (
                        set()
                    )  # To prevent infinite loops in rare cyclic exception references

                    while (
                        current_exc is not None
                        and id(current_exc) not in visited_exceptions
                    ):
                        visited_exceptions.add(id(current_exc))

                        if isinstance(current_exc, retry_exceptions):
                            should_retry = True
                            break

                        # First check __cause__ (from "raise X from Y")
                        if current_exc.__cause__ is not None:
                            current_exc = current_exc.__cause__
                        # Then check __context__ (from "try: ... except: raise X")
                        elif current_exc.__context__ is not None:
                            current_exc = current_exc.__context__
                        else:
                            # No more causes in the chain
                            break

                    if not should_retry:
                        # Not a retry exception or caused by a retry exception, so re-raise
                        raise

                    if attempt >= retries - 1:  # Last attempt
                        raise  # Re-raise the last exception

                    # Calculate exponential backoff with jitter
                    wait_time = backoff_factor * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt+1}/{retries}). "
                        f"Retrying in {wait_time:.2f}s. Error: {e!s}"
                    )
                    time.sleep(wait_time)

            raise ValueError("there was a problem") from None

        return wrapper

    return decorator


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class LRUCache:
    """An LRU (Least Recently Used) cache that stores a limited number of items.

    This cache automatically removes the least recently used item when it
    exceeds its max size. It behaves similarly to a dictionary, allowing
    items to be added and accessed using `[]` syntax.

    This implementation is thread-safe, using a lock to ensure that only one
    thread can modify or access the cache at any time.

    Args:
        max_size (int):
            The maximum number of items to store in the cache.
            Items exceeding this limit are automatically removed based on least
            recent usage.
    """

    def __init__(self, max_size=10):
        self._max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.Lock()  # Lock to ensure thread safety

    @property
    def max_size(self):
        with self._lock:
            return self._max_size

    @max_size.setter
    def max_size(self, size):
        with self._lock:
            self._max_size = size
            # Adjust the cache if the new size is smaller than the current number of items
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def __setitem__(self, key, value):
        with self._lock:
            # If the key already exists, remove it first to refresh its order
            if key in self._cache:
                self._cache.pop(key)

            # Add the new item to the cache (most recently used)
            self._cache[key] = value

            # If the cache exceeds the specified size, remove the least recently used item
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def __getitem__(self, key):
        with self._lock:
            if key in self._cache:
                # Move the accessed item to the end (mark as most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            raise KeyError(f"{key} not found in cache")

    def set(self, key, value):
        """Sets a key-value pair in the cache."""
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            self._cache[key] = value
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def get(self, key, default=None):
        """Gets a value from the cache by key, returning `default` if the key is not found."""
        with self._lock:
            if key in self._cache:
                value = self._cache.pop(key)
                self._cache[key] = value  # Move item to end to mark as recently used
                return value
            return default

    def __contains__(self, key):
        with self._lock:
            return key in self._cache

    def __len__(self):
        with self._lock:
            return len(self._cache)

    def __repr__(self):
        with self._lock:
            return f"LRUCache(max_size={self._max_size}, items={list(self._cache.items())})"


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


@lru_cache(maxsize=None)
def artifacts_json_cache(artifact_path):
    return load_json(artifact_path)


def load_json(path):
    with open(path) as f:
        try:
            return json.load(f)
        except json.decoder.JSONDecodeError as e:
            with open(path) as f:
                file_content = "\n".join(f.readlines())
            raise RuntimeError(
                f"Failed to decode json file at '{path}' with file content:\n{file_content}"
            ) from e


def save_to_file(path, data):
    with open(path, "w") as f:
        f.write(data)
        f.write("\n")


def json_dump(data):
    return json.dumps(data, indent=4, ensure_ascii=False)


def is_package_installed(package_name):
    """Check if a package is installed.

    Parameters:
    - package_name (str): The name of the package to check.

    Returns:
    - bool: True if the package is installed, False otherwise.
    """
    unitxt_pkg = importlib.util.find_spec(package_name)
    return unitxt_pkg is not None


def is_module_available(module_name):
    """Check if a module is available in the current Python environment.

    Parameters:
    - module_name (str): The name of the module to check.

    Returns:
    - bool: True if the module is available, False otherwise.
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def remove_numerics_and_quoted_texts(input_str):
    # Remove floats first to avoid leaving stray periods
    input_str = re.sub(r"\d+\.\d+", "", input_str)

    # Remove integers
    input_str = re.sub(r"\d+", "", input_str)

    # Remove strings in single quotes
    input_str = re.sub(r"'.*?'", "", input_str)

    # Remove strings in double quotes
    input_str = re.sub(r'".*?"', "", input_str)

    # Remove strings in triple quotes
    return re.sub(r'""".*?"""', "", input_str, flags=re.DOTALL)


def safe_eval(expression: str, context: dict, allowed_tokens: list) -> any:
    """Evaluates a given expression in a restricted environment, allowing only specified tokens and context variables.

    Args:
        expression (str): The expression to evaluate.
        context (dict): A dictionary mapping variable names to their values, which
                        can be used in the expression.
        allowed_tokens (list): A list of strings representing allowed tokens (such as
                               operators, function names, etc.) that can be used in the expression.

    Returns:
        any: The result of evaluating the expression.

    Raises:
        ValueError: If the expression contains tokens not in the allowed list or context keys.

    Note:
        This function should be used carefully, as it employs `eval`, which can
        execute arbitrary code. The function attempts to mitigate security risks
        by restricting the available tokens and not exposing built-in functions.
    """
    allowed_sub_strings = list(context.keys()) + allowed_tokens
    if is_made_of_sub_strings(
        remove_numerics_and_quoted_texts(expression), allowed_sub_strings
    ):
        return eval(expression, {"__builtins__": {}}, context)
    raise ValueError(
        f"The expression '{expression}' can not be evaluated because it contains tokens outside the allowed list of {allowed_sub_strings}."
    )


def import_module_from_file(file_path):
    # Get the module name (file name without extension)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    # Create a module specification
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    # Create a new module based on the specification
    module = importlib.util.module_from_spec(spec)
    # Load the module
    spec.loader.exec_module(module)
    return module


def deep_copy(obj):
    """Creates a deep copy of the given object.

    Args:
        obj: The object to be deep copied.

    Returns:
        A deep copy of the original object.
    """
    return copy.deepcopy(obj)


def shallow_copy(obj):
    """Creates a shallow copy of the given object.

    Args:
        obj: The object to be shallow copied.

    Returns:
        A shallow copy of the original object.
    """
    return copy.copy(obj)


def recursive_copy(obj, internal_copy=None):
    """Recursively copies an object with a selective copy method.

    For `list`, `dict`, and `tuple` types, it recursively copies their contents.
    For other types, it uses the provided `internal_copy` function if available.
    Objects without a `copy` method are returned as is.

    Args:
        obj: The object to be copied.
        internal_copy (callable, optional): The copy function to use for non-container objects.
            If `None`, objects without a `copy` method are returned as is.

    Returns:
        The recursively copied object.
    """
    # Handle dictionaries
    if isinstance(obj, dict):
        return type(obj)(
            {key: recursive_copy(value, internal_copy) for key, value in obj.items()}
        )

    # Handle named tuples
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(recursive_copy(item, internal_copy) for item in obj))

    # Handle tuples and lists
    if isinstance(obj, (tuple, list)):
        return type(obj)(recursive_copy(item, internal_copy) for item in obj)

    if internal_copy is None:
        return obj

    return internal_copy(obj)


def recursive_deep_copy(obj):
    """Performs a recursive deep copy of the given object.

    This function uses `deep_copy` as the internal copy method for non-container objects.

    Args:
        obj: The object to be deep copied.

    Returns:
        A recursively deep-copied version of the original object.
    """
    return recursive_copy(obj, deep_copy)


def recursive_shallow_copy(obj):
    """Performs a recursive shallow copy of the given object.

    This function uses `shallow_copy` as the internal copy method for non-container objects.

    Args:
        obj: The object to be shallow copied.

    Returns:
        A recursively shallow-copied version of the original object.
    """
    return recursive_copy(obj, shallow_copy)


class LongString(str):
    def __new__(cls, value, *, repr_str=None):
        obj = super().__new__(cls, value)
        obj._repr_str = repr_str
        return obj

    def __repr__(self):
        if self._repr_str is not None:
            return self._repr_str
        return super().__repr__()
