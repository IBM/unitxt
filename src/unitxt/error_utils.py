from typing import Optional

from .logging_utils import get_logger

logger = get_logger()


class Documentation:
    URL = "https://www.unitxt.ai/en/latest/"
    HUGGINGFACE_METRICS = "docs/adding_metric.html#adding-a-hugginface-metric"
    ADDING_TASK = "docs/adding_task.html"
    ADDING_TEMPLATE = "docs/adding_template.html"
    MULTIPLE_METRICS_OUTPUTS = (
        "docs/adding_metric.html#metric-outputs-with-multiple-metrics"
    )


def additional_info(path: str) -> str:
    return f"\nFor more information: see {Documentation.URL}/{path} \n"


class UnitxtError(Exception):
    """Exception raised for Unitxt errors.

    Attributes:
        message : str -- explanation of the error
        additional_info_id : Optional[str] -- relative path to additional documentation on web
        If set, should be one of the DOCUMENATION_* constants in the error_utils.py file.

    """

    def __init__(self, message: str, additional_info_id: Optional[str] = None):
        if additional_info_id is not None:
            message += additional_info(additional_info_id)
        super().__init__(message)


class UnitxtWarning:
    """Object to format warning message to log.

    Attributes:
        message -- explanation of the warning
        additional_info_id : Optional[str] -- relative path to additional documentation on web
        If set, should be one of the DOCUMENATION_* constants in the error_utils.py file.
    """

    def __init__(self, message: str, additional_info_id: Optional[str] = None):
        if additional_info_id is not None:
            message += additional_info(additional_info_id)
        logger.warning(message)
