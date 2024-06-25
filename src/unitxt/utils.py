import importlib.util
import json
import os
from functools import lru_cache
from typing import Any, Dict

import pkg_resources

from .text_utils import is_made_of_sub_strings


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


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
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False


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
    if is_made_of_sub_strings(expression, allowed_sub_strings):
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
