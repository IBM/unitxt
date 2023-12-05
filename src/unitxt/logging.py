import logging
import os
import sys
import threading
from typing import Optional

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "detail": logging.DEBUG,  # will also print filename and line number
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING


def setup_logging():
    logging.basicConfig(
        format="%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=_default_log_level,
    )


def _get_default_logging_level():
    env_level_str = os.getenv("UNITXT_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]

        logging.getLogger().warning(
            f"Unknown option UNITXT_VERBOSITY={env_level_str}, "
            f"has to be one of: { ', '.join(log_levels.keys()) }"
        )
    return _default_log_level


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(__name__.split(".")[0])


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler is not None:
            return
        _default_handler = logging.StreamHandler()
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler.flush = sys.stderr.flush

        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())

        if os.getenv("UNITXT_VERBOSITY", None) == "detail":
            formatter = logging.Formatter(
                "[Unitxt|%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s"
            )
            _default_handler.setFormatter(formatter)

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
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.Formatter(
            "[Unitxt|%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
        )
        handler.setFormatter(formatter)
