import re
from contextlib import contextmanager
from typing import Any, Optional

from .logging_utils import get_logger
from .settings_utils import get_constants

constants = get_constants()
logger = get_logger()


class Documentation:
    URL = "https://www.unitxt.ai/en/latest/"
    HUGGINGFACE_METRICS = "docs/adding_metric.html#adding-a-hugginface-metric"
    ADDING_TASK = "docs/adding_task.html"
    ADDING_TEMPLATE = "docs/adding_template.html"
    POST_PROCESSORS = "docs/adding_template.html#post-processors"
    MULTIPLE_METRICS_OUTPUTS = (
        "docs/adding_metric.html#metric-outputs-with-multiple-metrics"
    )
    EVALUATION = "docs/evaluating_datasets.html"
    BENCHMARKS = "docs/benchmark.html"
    DATA_CLASSIFICATION_POLICY = "docs/data_classification_policy.html"
    CATALOG = "docs/saving_and_loading_from_catalog.html"
    SETTINGS = "docs/settings.html"


def additional_info(path: str) -> str:
    return f"\nFor more information: see {Documentation.URL}/{path} \n"


class UnitxtError(Exception):
    """Exception raised for Unitxt errors.

    Args:
        message (str): explanation of the error
        additional_info_id (Optional[str]): relative path to additional documentation on web
            If set, should be one of the DOCUMENTATION_* constants in the error_utils.py file.
    """

    def __init__(self, message: str, additional_info_id: Optional[str] = None):
        if additional_info_id is not None:
            message += additional_info(additional_info_id)
        super().__init__(message)


class UnitxtWarning:
    """Object to format warning message to log.

    Args:
        message (str): explanation of the warning
        additional_info_id (Optional[str]): relative path to additional documentation on web
            If set, should be one of the DOCUMENTATION_* constants in the error_utils.py file.
    """

    def __init__(self, message: str, additional_info_id: Optional[str] = None):
        if additional_info_id is not None:
            message += additional_info(additional_info_id)
        logger.warning(message)


context_block_title = "🦄 Unitxt Error Context"


def _visible_length(text: str) -> int:
    import unicodedata

    ansi_escape = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\]8;;[^\x1b]*\x1b\\")
    clean_text = ansi_escape.sub("", text)
    width = 0
    for char in clean_text:
        if (
            unicodedata.east_asian_width(char) in ("F", "W")
            or 0x1F300 <= ord(char) <= 0x1F9FF
        ):
            width += 2
        else:
            width += 1
    return width


def _make_object_clickable(
    full_obj_name: str, display_name: Optional[str] = None
) -> str:
    import os

    if display_name is None:
        display_name = full_obj_name.split(".")[-1]
    if full_obj_name.startswith("unitxt."):
        parts = full_obj_name.split(".")
        if len(parts) >= 2:
            module_path = ".".join(parts[:2])
            doc_url = f"{Documentation.URL}{module_path}.html#{full_obj_name}"
            if (
                os.environ.get("TERM_PROGRAM") in ["iTerm.app", "vscode"]
                or os.environ.get("TERMINAL_EMULATOR") == "JetBrains-JediTerm"
            ):
                return f"\033]8;;{doc_url}\033\\{display_name}\033]8;;\033\\"
            return f"{display_name} ({doc_url})"
    return display_name


def _get_existing_context(error: Exception):
    """Extract existing context from an error if it exists."""
    if hasattr(error, "__error_context__"):
        existing = error.__error_context__
        return (
            existing["original_message"],
            existing["context_object"],
            existing["context"],
        )
    return str(error), None, {}


def _format_object_context(obj: Any) -> Optional[str]:
    """Format an object for display in error context."""
    if obj is None:
        return None
    if hasattr(obj, "__class__"):
        class_name = obj.__class__.__name__
        module_name = getattr(obj.__class__, "__module__", "")
    else:
        obj_type = type(obj)
        class_name = obj_type.__name__
        module_name = getattr(obj_type, "__module__", "")
    if module_name:
        full_name = f"{module_name}.{class_name}"
        clickable_object = _make_object_clickable(full_name, class_name)
        return f"Object: {clickable_object}"
    return f"Object: {class_name}"


def _make_clickable_link(url: str) -> str:
    """Create a clickable terminal link."""
    import os

    if (
        os.environ.get("TERM_PROGRAM") in ["iTerm.app", "vscode"]
        or os.environ.get("TERMINAL_EMULATOR") == "JetBrains-JediTerm"
    ):
        return f"\033]8;;{url}\033\\link\033]8;;\033\\"
    return url


def _format_help_context(help_docs) -> list:
    """Format help documentation into context parts."""
    parts = []
    if isinstance(help_docs, str):
        parts.append(f"Help: {_make_clickable_link(help_docs)}")
    elif isinstance(help_docs, dict):
        for label, url in help_docs.items():
            parts.append(f"Help ({label}): {_make_clickable_link(url)}")
    elif isinstance(help_docs, list):
        for item in help_docs:
            if isinstance(item, dict) and len(item) == 1:
                label, url = next(iter(item.items()))
                parts.append(f"Help ({label}): {_make_clickable_link(url)}")
            elif isinstance(item, str):
                parts.append(f"Help: {_make_clickable_link(item)}")
    return parts


def _build_context_parts(context_object: Any, context: dict) -> list:
    """Build the list of context information parts."""
    parts = []
    ordered_keys = [
        "Python",
        "Unitxt",
        "Stage",
        "Stream",
        "Index",
        "Instance",
        "Object",
        "Action",
    ]
    processed_keys = set()

    for desired_key in ordered_keys:
        for actual_key in context.keys():
            if actual_key.lower() == desired_key.lower():
                value = (
                    "unknown" if context[actual_key] is None else context[actual_key]
                )
                parts.append(f"{actual_key.replace('_', ' ').title()}: {value}")
                processed_keys.add(actual_key)
                break

    if not any(key.lower() == "object" for key in processed_keys):
        obj_context = _format_object_context(context_object)
        if obj_context:
            parts.append(obj_context)

    processed_keys.add("help")
    for key, value in context.items():
        if key not in processed_keys:
            value = "unknown" if value is None else value
            parts.append(f"{key.replace('_', ' ').title()}: {value}")

    if "help" in context:
        parts.extend(_format_help_context(context["help"]))
    else:
        parts.append(f"Help: {_make_clickable_link(Documentation.URL)}")

    return parts


def _create_context_box(parts: list) -> str:
    """Create a formatted box containing context information."""
    if not parts:
        return ""
    max_width = (
        max(
            _visible_length(context_block_title),
            max(_visible_length(part) for part in parts),
        )
        + 4
    )
    top_line = "┌" + "─" * max_width + "┐"
    bottom_line = "└" + "─" * max_width + "┘"
    lines = [top_line]
    lines.append(
        f"│ {context_block_title}{' ' * (max_width - _visible_length(context_block_title) - 1)}│"
    )
    lines.append(f"│ {'-' * (max_width - 2)} │")
    for part in parts:
        padding = " " * (max_width - _visible_length(part) - 4)
        lines.append(f"│  - {part}{padding}│")
    lines.append(bottom_line)
    return "\n".join(lines)


def _store_context_attributes(
    error: Exception, context_object: Any, context: dict, original_message: str
):
    """Store context information in error attributes."""
    error.__error_context__ = {
        "context_object": context_object,
        "context": context,
        "original_message": original_message,
    }
    try:
        error.original_error = type(error)(original_message)
    except (TypeError, ValueError):
        error.original_error = Exception(original_message)
    error.context_object = context_object
    error.context = context


def _add_context_to_exception(
    original_error: Exception, context_object: Any = None, **context
):
    """Add context information to an exception by modifying its message."""
    original_message, existing_object, existing_context = _get_existing_context(
        original_error
    )
    final_context_object = existing_object or context_object
    final_context = {
        "Unitxt": constants.version,
        "Python": constants.python,
        **existing_context,
        **context,
    }
    context_parts = _build_context_parts(final_context_object, final_context)
    context_message = _create_context_box(context_parts)
    _store_context_attributes(
        original_error, final_context_object, final_context, original_message
    )
    if context_parts:
        formatted_message = f"\n{context_message}\n\n{original_message}"
        original_error.args = (formatted_message,)
    else:
        original_error.args = (original_message,)


@contextmanager
def error_context(context_object: Any = None, **context):
    """Context manager that catches exceptions and re-raises them with additional context.

    Args:
        context_object: The object being processed (optional)
        **context: Any additional context to include in the error message.
                  You can provide any key-value pairs that help identify where the error occurred.

                  Special context keys:
                  - help: Documentation links to help with the error.
                    Can be a string (single URL), dict (label: URL), or list of URLs/dicts.

    Examples:
        with error_context(self, operation="validation", item_id=42):
            result = process_item(item)

        with error_context(operation="schema_validation", help="https://docs.example.com/schema"):
            validate_schema(data)

        with error_context(processor, step="preprocessing", batch_size=32):
            results = process_batch(batch)
    """
    try:
        yield
    except Exception as e:
        _add_context_to_exception(e, context_object, **context)
        raise
