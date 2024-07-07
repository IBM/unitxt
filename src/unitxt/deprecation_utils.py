import functools
import warnings

from .settings_utils import get_constants, get_settings

constants = get_constants()
settings = get_settings()


class DeprecationError(Exception):
    """Custom exception for deprecated versions."""

    pass


def compare_versions(version1, version2):
    """Compare two semantic versioning strings and determine their relationship.

    Parameters:
    - version1 (str): The first version string to compare.
    - version2 (str): The second version string to compare.

    Returns:
    - int: -1 if version1 < version2, 1 if version1 > version2, 0 if equal.

    Example:
    >>> compare_versions("1.2.0", "1.2.3")
    -1
    >>> compare_versions("1.3.0", "1.2.8")
    1
    >>> compare_versions("1.0.0", "1.0.0")
    0
    """
    parts1 = [int(part) for part in version1.split(".")]
    parts2 = [int(part) for part in version2.split(".")]
    length_difference = len(parts1) - len(parts2)
    if length_difference > 0:
        parts2.extend([0] * length_difference)
    elif length_difference < 0:
        parts1.extend([0] * (-length_difference))
    for part1, part2 in zip(parts1, parts2):
        if part1 < part2:
            return -1
        if part1 > part2:
            return 1
    return 0


def depraction_wrapper(obj, version, alt_text):
    """A wrapper function for deprecation handling, issuing warnings or errors based on version comparison.

    Args:
        obj (callable): The object to be wrapped, typically a function or class method.
        version (str): The version at which the object becomes deprecated.
        alt_text (str): Additional text to display, usually suggests an alternative.

    Returns:
        callable: A wrapped version of the original object that checks for deprecation.
    """

    @functools.wraps(obj)
    def wrapper(*args, **kwargs):
        if constants.version < version:
            if settings.default_verbosity in ["debug", "info", "warning"]:
                warnings.warn(
                    f"{obj.__name__} is deprecated.{alt_text}",
                    DeprecationWarning,
                    stacklevel=2,
                )
        elif constants.version >= version:
            raise DeprecationError(f"{obj.__name__} is no longer supported.{alt_text}")
        return obj(*args, **kwargs)

    return wrapper


def deprecation(version, alternative=None):
    """Decorator for marking functions or class methods as deprecated.

    Args:
        version (str): The version at which the function or method becomes deprecated.
        alternative (str, optional): Suggested alternative to the deprecated functionality.

    Returns:
        callable: A decorator that can be applied to functions or class methods.
    """

    def decorator(obj):
        alt_text = f" Use {alternative} instead." if alternative is not None else ""
        if callable(obj):
            func = obj
        elif hasattr(obj, "__init__"):
            func = obj.__init__
        else:
            raise ValueError("Unsupported object type for deprecation.")
        return depraction_wrapper(func, version, alt_text)

    return decorator
