import logging
import os
import sys
import threading
from typing import Optional

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.DEBUG


def _get_default_logging_level():
    env_level_str = os.getenv("UNITXT_VERBOSITY", None)
    if env_level_str is not None:
        try:
            return log_levels[env_level_str]
        except KeyError as e:
            raise ValueError(
                f"UNITXT_VERBOSITY has to be one of: { ', '.join(log_levels.keys()) }. Got {env_level_str}."
            ) from e
    return _default_log_level


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(__name__.split(".")[0])


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler is not None:
            return
        _default_handler = logging.StreamHandler(sys.stdout)
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler.flush = sys.stderr.flush

        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())

        library_root_logger.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        name = __name__.split(".")[0]

    _configure_library_root_logger()
    return logging.getLogger(name)


def set_verbosity(level):
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(log_levels.get(level))


def enable_explicit_format() -> None:
    for handler in _get_library_root_logger().handlers:
        formatter = logging.Formatter(
            "[Unitxt|%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
        )
        handler.setFormatter(formatter)
