from contextlib import contextmanager
from typing import Any, Optional

from scipy import constants

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
        message (str):
            explanation of the error
        additional_info_id (Optional[str]):
            relative path to additional documentation on web
            If set, should be one of the DOCUMENATION_* constants in the error_utils.py file.

    """

    def __init__(self, message: str, additional_info_id: Optional[str] = None):
        if additional_info_id is not None:
            message += additional_info(additional_info_id)
        super().__init__(message)


class UnitxtWarning:
    """Object to format warning message to log.

    Args:
        message (str):
            explanation of the warning
        additional_info_id (Optional[str]):
            relative path to additional documentation on web
            If set, should be one of the DOCUMENATION_* constants in the error_utils.py file.
    """

    def __init__(self, message: str, additional_info_id: Optional[str] = None):
        if additional_info_id is not None:
            message += additional_info(additional_info_id)
        logger.warning(message)


context_block_title = "Unitxt Error Context"


def _add_context_to_exception(
    original_error: Exception, context_object: Any = None, **context
):
    """Add context information to an exception by modifying its message."""
    # Get the original message, removing any existing context
    original_message = str(original_error)
    if hasattr(original_error, "__error_context__"):
        # If this error already has context, get the original message from the stored context
        original_message = original_error.__error_context__["original_message"]
        context = {
            "Unitxt": constants.version,
            "Python": constants.python,
            **original_error.__error_context__["context"],
            **context,
        }
    # Build context message from all provided context information
    context_parts = []

    # Add object context if provided
    if context_object is not None:
        if hasattr(context_object, "__class__"):
            context_parts.append(f"Object: {context_object.__class__.__name__}")
        else:
            context_parts.append(f"Object: {type(context_object).__name__}")

    for key in context.keys():
        value = context[key]
        if value is None:
            value = "unknown"
        context_parts.append(f"{key.replace('_', ' ').title()}: {value}")

    # Store context message for display
    if context_parts:
        # Create a box around the context information
        max_width = (
            max(len(context_block_title), max(len(part) for part in context_parts)) + 4
        )
        top_line = "┌" + "─" * max_width + "┐"
        bottom_line = "└" + "─" * max_width + "┘"

        context_lines = [top_line]
        context_lines.append(
            f"│ {context_block_title}{' ' * (max_width - len(context_block_title) - 1)}│"
        )
        for part in context_parts:
            context_lines.append(f"│  - {part}{' ' * (max_width - len(part) - 4)}│")
        context_lines.append(bottom_line)

        context_message = "\n" + "\n".join(context_lines)
        enhanced_message = f"{context_message}\n\n{original_message}"
    else:
        enhanced_message = original_message

    # Store context info in a special attribute
    original_error.__error_context__ = {
        "context_object": context_object,
        "context": context,
        "original_message": original_message,
    }

    # Store original information as attributes for backward compatibility
    try:
        original_error.original_error = type(original_error)(original_message)
    except (TypeError, ValueError):
        # For custom exceptions with complex constructors, create a basic Exception
        original_error.original_error = Exception(original_message)

    original_error.context_object = context_object
    original_error.context = context

    # Simply modify the exception's args to include the enhanced message
    original_error.args = (enhanced_message,)

    # Handle KeyError's special __str__ method that adds quotes
    if isinstance(original_error, KeyError):

        def enhanced_str(self):
            return enhanced_message

        # Bind the method to the instance
        import types

        original_error.__str__ = types.MethodType(enhanced_str, original_error)


@contextmanager
def error_context(context_object: Any = None, **context):
    """Context manager that catches exceptions and re-raises them with additional context.

    Args:
        context_object: The object being processed (optional)
        **context: Any additional context to include in the error message.
                  You can provide any key-value pairs that help identify where the error occurred.

    Examples:
        # Basic usage with object and context
        with error_context(self, operation="validation", item_id=42):
            result = process_item(item)

        # File processing context
        with error_context(input_file="data.json", line_number=156):
            data = parse_line(line)

        # Processing context
        with error_context(processor, step="preprocessing", batch_size=32):
            results = process_batch(batch)

        # Database context
        with error_context(db_connection, table="users", operation="SELECT"):
            results = execute_query(query)
    """
    try:
        yield
    except Exception as e:
        # Add context to the original exception by modifying its message
        _add_context_to_exception(e, context_object, **context)
        # Re-raise the same exception - this preserves the original traceback completely
        raise
