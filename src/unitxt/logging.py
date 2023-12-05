"""Logging utilities."""


import functools
import logging
import os
import sys
import threading
from logging import (
    CRITICAL,  # NOQA
    DEBUG,
    ERROR,
    FATAL,  # NOQA
    INFO,
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,
)
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

_tqdm_active = True


def setup_logging():
    logging.basicConfig(
        format="%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=_default_log_level,
    )


def _get_default_logging_level():
    """If UNITXT_VERBOSITY env var is set to one of the valid choices return that as the new default level.

    If it is
    not - fall back to `_default_log_level`.
    """
    env_level_str = os.getenv("UNITXT_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]

        logging.getLogger().warning(
            f"Unknown option UNITXT_VERBOSITY={env_level_str}, "
            f"has to be one of: { ', '.join(log_levels.keys()) }"
        )
    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        # set defaults based on https://github.com/pyinstaller/pyinstaller/issues/7334#issuecomment-1357447176
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        # if logging level is debug, we add pathname and lineno to formatter for easy debugging
        if os.getenv("UNITXT_VERBOSITY", None) == "detail":
            formatter = logging.Formatter(
                "[%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s"
            )
            _default_handler.setFormatter(formatter)

        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_log_levels_dict():
    return log_levels


def capture_warnings(capture):
    """Calls the `captureWarnings` method from the logging library to enable management of the warnings emitted by the `warnings` library.

    Read more about this method here:
    https://docs.python.org/3/library/logging.html#integration-with-the-warnings-module

    All warnings will be logged through the `py.warnings` logger.

    Careful: this method also adds a handler to this logger if it does not already have one, and updates the logging
    level of that logger to the library's root logger.
    """
    logger = get_logger("py.warnings")

    if not logger.handlers:
        logger.addHandler(_default_handler)

    logger.setLevel(_get_library_root_logger().level)

    capture_warnings(capture)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom unitxt module.
    """
    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    """Return the current level for the unitxt's root logger as an int.

    Returns:
        `int`: The logging level.

    <Tip>

    🤗 Transformers has following logging levels:

    - 50: `unitxt.logging.CRITICAL` or `unitxt.logging.FATAL`
    - 40: `unitxt.logging.ERROR`
    - 30: `unitxt.logging.WARNING` or `unitxt.logging.WARN`
    - 20: `unitxt.logging.INFO`
    - 10: `unitxt.logging.DEBUG`

    </Tip>
    """
    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """Set the verbosity level for the unitxt's root logger.

    Args:
        verbosity (`int`):
            Logging level, e.g., one of:

            - `unitxt.logging.CRITICAL` or `unitxt.logging.FATAL`
            - `unitxt.logging.ERROR`
            - `unitxt.logging.WARNING` or `unitxt.logging.WARN`
            - `unitxt.logging.INFO`
            - `unitxt.logging.DEBUG`
    """
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    """Set the verbosity to the `INFO` level."""
    return set_verbosity(INFO)


def set_verbosity_warning():
    """Set the verbosity to the `WARNING` level."""
    return set_verbosity(WARNING)


def set_verbosity_debug():
    """Set the verbosity to the `DEBUG` level."""
    return set_verbosity(DEBUG)


def set_verbosity_error():
    """Set the verbosity to the `ERROR` level."""
    return set_verbosity(ERROR)


def disable_default_handler() -> None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def add_handler(handler: logging.Handler) -> None:
    """Adds a handler to the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()

    assert handler is not None
    _get_library_root_logger().addHandler(handler)


def remove_handler(handler: logging.Handler) -> None:
    """Removes given handler from the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()

    assert handler is not None and handler not in _get_library_root_logger().handlers
    _get_library_root_logger().removeHandler(handler)


def disable_propagation() -> None:
    """Disable propagation of the library log outputs. Note that log propagation is disabled by default."""
    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """Enable propagation of the library log outputs.

    Please disable the HuggingFace Transformers's default handler to
    prevent double logging if the root logger has been configured.
    """
    _configure_library_root_logger()
    _get_library_root_logger().propagate = True


def enable_explicit_format() -> None:
    """Enable explicit formatting for every HuggingFace Transformers's logger.

    The explicit formatter is as follows:
    ```
        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE
    ```
    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.Formatter(
            "[Unitxt|%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
        )
        handler.setFormatter(formatter)


def reset_format() -> None:
    """Resets the formatting for HuggingFace Transformers's loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        handler.setFormatter(None)


def warning_advice(self, *args, **kwargs):
    """This method is identical to `logger.warning()`, but if env var UNITXT_NO_ADVISORY_WARNINGS=1 is set, this warning will not be printed."""
    no_advisory_warnings = os.getenv("UNITXT_NO_ADVISORY_WARNINGS", False)
    if no_advisory_warnings:
        return
    self.warning(*args, **kwargs)


logging.Logger.warning_advice = warning_advice


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """This method is identical to `logger.warning()`, but will emit the warning with the same message only once.

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once
