import logging
import os
import sys
import threading
from typing import Optional

from .settings_utils import get_settings

settings = get_settings()

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def _get_default_logging_level():
    try:
        return log_levels[settings.default_verbosity]
    except KeyError as e:
        raise ValueError(
            f"unitxt.settings.default_verobsity or env variable UNITXT_DEFAULT_VERBOSITY has to be one of: { ', '.join(log_levels.keys()) }. Got {settings.default_verbosity}."
        ) from e


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(__name__.split(".")[0])


class SizeLimitedFormatter(logging.Formatter):
    def format(self, record):
        original_message = super().format(record)
        max_size = settings.max_log_message_size
        if len(original_message) > max_size:
            return (
                original_message[:max_size]
                + f"...\n(Message is too long > {max_size}. Can be set through unitxt.settings.max_log_message_size or UNITXT_MAX_LOG_MESSAGE_SIZE environment variable.)"
            )
        return original_message


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler is not None:
            return
        _default_handler = logging.StreamHandler(sys.stdout)
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler.flush = sys.stderr.flush
        _default_handler.setFormatter(SizeLimitedFormatter())

        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())

        library_root_logger.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        name = __name__.split(".")[0]

    _configure_library_root_logger()
    return logging.getLogger(name)


settings._logger = get_logger("settings")


def set_verbosity(level):
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(log_levels.get(level))


def enable_explicit_format() -> None:
    for handler in _get_library_root_logger().handlers:
        formatter = SizeLimitedFormatter(
            "[Unitxt|%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
        )
        handler.setFormatter(formatter)
