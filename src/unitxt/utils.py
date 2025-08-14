import copy
import functools
import importlib.util
import inspect
import json
import os
import random
import re
import time
import types
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_installed_version
from typing import Any, Dict, Optional
from urllib.error import HTTPError as UrllibHTTPError

from packaging.requirements import Requirement
from packaging.version import Version
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
    def __init__(self, max_size: Optional[int] = 10):
        self._max_size = max_size
        self._context_cache = ContextVar("context_lru_cache", default=None)

    def _get_cache(self):
        cache = self._context_cache.get()
        if cache is None:
            cache = OrderedDict()
            self._context_cache.set(cache)
        return cache

    def __setitem__(self, key, value):
        cache = self._get_cache()
        if key in cache:
            cache.pop(key)
        cache[key] = value
        if self._max_size is not None:
            while len(cache) > self._max_size:
                cache.popitem(last=False)

    def __getitem__(self, key):
        cache = self._get_cache()
        if key in cache:
            value = cache.pop(key)
            cache[key] = value
            return value
        raise KeyError(f"{key} not found in cache")

    def get(self, key, default=None):
        cache = self._get_cache()
        if key in cache:
            value = cache.pop(key)
            cache[key] = value
            return value
        return default

    def clear(self):
        """Clear all items from the cache."""
        cache = self._get_cache()
        cache.clear()

    def __contains__(self, key):
        return key in self._get_cache()

    def __len__(self):
        return len(self._get_cache())

    def __repr__(self):
        return f"LRUCache(max_size={self._max_size}, items={list(self._get_cache().items())})"


def lru_cache_decorator(max_size=128):
    def decorator(func):
        cache = LRUCache(max_size=max_size)

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = args
            if kwargs:
                key += tuple(sorted(kwargs.items()))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result

        wrapper.cache_clear = cache.clear
        return wrapper

    return decorator


@lru_cache_decorator(max_size=None)
def artifacts_json_cache(artifact_path):
    return load_json(artifact_path)


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


def load_json(path):
    with open(path) as f:
        try:
            return json.load(f, object_hook=decode_function)
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


def encode_function(obj):
    # Allow only plain (module-level) functions
    if isinstance(obj, types.FunctionType):
        try:
            return {"__function__": obj.__name__, "source": get_function_source(obj)}
        except Exception as e:
            raise TypeError(f"Failed to serialize function {obj.__name__}") from e
    elif isinstance(obj, types.MethodType):
        raise TypeError(
            f"Method {obj.__func__.__name__} of class {obj.__self__.__class__.__name__} is not JSON serializable"
        )
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def json_dump(data, sort_keys=False):
    return json.dumps(
        data, indent=4, default=encode_function, ensure_ascii=False, sort_keys=sort_keys
    )


def get_function_source(func):
    if hasattr(func, "__exec_source__"):
        return func.__exec_source__
    return inspect.getsource(func)


def decode_function(obj):
    # Detect our special function marker
    if "__function__" in obj and "source" in obj:
        namespace = {}
        func_name = obj["__function__"]
        try:
            exec(obj["source"], namespace)
            func = namespace.get(func_name)
            func.__exec_source__ = obj["source"]
            if not callable(func):
                raise ValueError(
                    f"Source did not define a callable named {func_name!r}"
                )
            return func
        except Exception as e:
            raise ValueError(
                f"Failed to load function {func_name!r} from source:\n{obj['source']}"
            ) from e

    return obj


def json_load(s):
    return json.loads(s, object_hook=decode_function)


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


class DistributionNotFound(Exception):
    def __init__(self, requirement):
        self.requirement = requirement
        super().__init__(f"Distribution not found for requirement: {requirement}")


class VersionConflict(Exception):
    def __init__(self, dist, req):
        self.dist = dist  # Distribution object, just emulate enough for your needs
        self.req = req
        super().__init__(f"Version conflict: {dist} does not satisfy {req}")


class DistStub:
    # Minimal stub to mimic pkg_resources.Distribution
    def __init__(self, project_name, version):
        self.project_name = project_name
        self.version = version


def require(requirements):
    """Minimal drop-in replacement for pkg_resources.require.

    Accepts a single requirement string or a list of them.
    Raises DistributionNotFound or VersionConflict.
    Returns nothing (side-effect only).
    """
    if isinstance(requirements, str):
        requirements = [requirements]
    for req_str in requirements:
        req = Requirement(req_str)
        if req.marker and not req.marker.evaluate():
            continue  # skip not needed for this environment
        name = req.name
        try:
            ver = get_installed_version(name)
        except PackageNotFoundError as e:
            raise DistributionNotFound(req_str) from e
        if req.specifier and not req.specifier.contains(Version(ver), prereleases=True):
            dist = DistStub(name, ver)
            raise VersionConflict(dist, req_str)
