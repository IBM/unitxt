import asyncio
import importlib.metadata
import importlib.util
import os
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar

from .version import version


def cast_to_type(value, value_type):
    if value_type is bool:
        if value not in ["True", "False", True, False]:
            raise ValueError(
                f"Value must be in ['True', 'False', True, False] got {value}"
            )
        if value == "True":
            return True
        if value == "False":
            return False
        return value
    if value_type is int:
        return int(value)
    if value_type is float:
        return float(value)

    raise ValueError("Unsupported type.")


class Settings:
    _instance = None
    _settings = {}
    _types = {}
    _logger = None
    _thread_local = threading.local()
    _context_settings = ContextVar("settings", default=None)

    @classmethod
    def is_uninitilized(cls):
        return cls._instance is None

    def __new__(cls):
        if cls.is_uninitilized():
            cls._instance = super().__new__(cls)
        return cls._instance

    def _is_async_context(self):
        """Check if we're in an async context."""
        try:
            asyncio.current_task()
            return True
        except RuntimeError:
            return False

    def _get_context_stack(self):
        """Get the current context stack (list of dicts)."""
        if self._is_async_context():
            stack = self._context_settings.get()
            return stack if stack is not None else []
        if not hasattr(self._thread_local, "stack"):
            self._thread_local.stack = []
        return self._thread_local.stack

    def __setattr__(self, key, value):
        if key.endswith("_key") or key in {"_instance", "_settings"}:
            raise AttributeError(f"Modifying '{key}' is not allowed.")

        if isinstance(value, tuple) and len(value) == 2:
            value_type, value = value
            if value_type not in [int, float, bool]:
                raise ValueError(
                    f"Setting settings with tuple requires the first element to be either [int, float, bool], got {value_type}"
                )
            self._types[key] = value_type

        if key in self._types and value is not None:
            value_type = self._types[key]
            value = cast_to_type(value, value_type)

        # Check if we're in a context
        stack = self._get_context_stack()
        if stack:
            # Modify the innermost context
            stack[-1][key] = value
            if self._logger is not None:
                self._logger.info(
                    f"unitxt.settings.{key} (context-local) changed to: {value}"
                )
        else:
            # Modify global settings
            if key in self._settings:
                if self._logger is not None:
                    self._logger.info(
                        f"unitxt.settings.{key} changed: {self._settings[key]} -> {value}"
                    )
            self._settings[key] = value

    def __getattr__(self, key):
        if key.endswith("_key"):
            actual_key = key[:-4]
            return self.environment_variable_key_name(actual_key)

        key_name = self.environment_variable_key_name(key)
        env_value = os.getenv(key_name)

        if env_value is not None:
            if key in self._types:
                env_value = cast_to_type(env_value, self._types[key])
            return env_value

        # Check context stack from innermost to outermost
        stack = self._get_context_stack()
        for context in reversed(stack):
            if key in context:
                return context[key]

        # Then check global settings
        if key in self._settings:
            return self._settings[key]

        raise AttributeError(f"'{key}' not found")

    def environment_variable_key_name(self, key):
        return "UNITXT_" + key.upper()

    def get_all_environment_variables(self):
        return [
            self.environment_variable_key_name(key) for key in self._settings.keys()
        ]

    @contextmanager
    def context(self, **kwargs):
        """Context manager that uses thread-local or async-local storage with proper nesting."""
        # Apply type conversion
        for key, value in kwargs.items():
            if key in self._types and value is not None:
                kwargs[key] = cast_to_type(value, self._types[key])

        if self._is_async_context():
            # Handle async context
            current_stack = self._context_settings.get()
            if current_stack is None:
                current_stack = []

            # Create new stack with added context
            new_stack = [*current_stack, kwargs.copy()]
            token = self._context_settings.set(new_stack)

            try:
                yield
            finally:
                self._context_settings.reset(token)
        else:
            # Handle thread-local context
            if not hasattr(self._thread_local, "stack"):
                self._thread_local.stack = []

            self._thread_local.stack.append(kwargs.copy())
            try:
                yield
            finally:
                self._thread_local.stack.pop()


class Constants:
    _instance = None
    _constants = {}

    @classmethod
    def is_uninitilized(cls):
        return cls._instance is None

    def __new__(cls):
        if cls.is_uninitilized():
            cls._instance = super().__new__(cls)
        return cls._instance

    def __setattr__(self, key, value):
        if key.endswith("_key") or key in {"_instance", "_constants"}:
            raise AttributeError(f"Modifying '{key}' is not allowed.")
        if key in self._constants:
            raise ValueError("Cannot override constants.")
        self._constants[key] = value

    def __getattr__(self, key):
        if key in self._constants:
            return self._constants[key]

        raise AttributeError(f"'{key}' not found")


if Settings.is_uninitilized():
    settings = Settings()
    settings.allow_unverified_code = (bool, False)
    settings.use_only_local_catalogs = (bool, False)
    settings.global_loader_limit = (int, None)
    settings.num_resamples_for_instance_metrics = (int, 1000)
    settings.num_resamples_for_global_metrics = (int, 100)
    settings.max_log_message_size = (int, 100000)
    settings.catalogs = None
    settings.artifactories = None
    settings.default_recipe = "dataset_recipe"
    settings.default_verbosity = "info"
    settings.use_eager_execution = False
    settings.remote_metrics = []
    settings.test_card_disable = (bool, False)
    settings.test_metric_disable = (bool, False)
    settings.metrics_master_key_token = None
    settings.seed = (int, 42)
    settings.skip_artifacts_prepare_and_verify = (bool, False)
    settings.data_classification_policy = None
    settings.mock_inference_mode = (bool, False)
    settings.disable_hf_datasets_cache = (bool, False)
    settings.stream_hf_datasets_by_default = (bool, False)
    settings.loader_cache_size = (int, 25)
    settings.loaders_max_retries = (int, 10)
    settings.task_data_as_text = (bool, True)
    settings.default_provider = "watsonx"
    settings.default_format = None
    settings.hf_offline_datasets_path = None
    settings.hf_offline_metrics_path = None
    settings.hf_offline_models_path = None
    settings.inference_engine_cache_path = "./inference_engine_cache/"
    settings.max_connection_retries = 3

if Constants.is_uninitilized():
    constants = Constants()
    constants.dataset_file = os.path.join(os.path.dirname(__file__), "dataset.py")
    constants.metric_file = os.path.join(os.path.dirname(__file__), "metric.py")
    constants.local_catalog_path = os.path.join(os.path.dirname(__file__), "catalog")
    unitxt_pkg = importlib.util.find_spec("unitxt")
    if unitxt_pkg and unitxt_pkg.origin:
        constants.package_dir = os.path.dirname(unitxt_pkg.origin)
        constants.default_catalog_path = os.path.join(constants.package_dir, "catalog")
    else:
        constants.default_catalog_path = constants.local_catalog_path
    constants.catalog_dir = constants.local_catalog_path
    constants.dataset_url = "unitxt/data"
    constants.metric_url = "unitxt/metric"
    constants.version = version
    constants.python = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    constants.catalog_hierarchy_sep = "."
    constants.env_local_catalogs_paths_sep = ":"
    constants.non_registered_files = [
        "__init__.py",
        "artifact.py",
        "utils.py",
        "register.py",
        "metric.py",
        "dataset.py",
        "blocks.py",
    ]
    constants.codebase_url = "https://github.com/IBM/unitxt"
    constants.website_url = "https://www.unitxt.org"
    constants.inference_stream = "__INFERENCE_STREAM__"
    constants.instance_stream = "__INSTANCE_STREAM__"
    constants.image_tag = "unitxt-img"
    constants.demos_pool_field = "_demos_pool_"
    constants.demos_field = "demos"
    constants.instruction_field = "instruction"
    constants.system_prompt_field = "system_prompt"


def get_settings() -> Settings:
    return Settings()


def get_constants():
    return Constants()
