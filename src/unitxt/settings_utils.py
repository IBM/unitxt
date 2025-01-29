"""Library Settings and Constants.

This module provides a mechanism for managing application-wide configuration and immutable constants. It includes the `Settings` and `Constants` classes, which are implemented as singleton patterns to ensure a single shared instance across the application. Additionally, it defines utility functions to access these objects and configure application behavior.

### Key Components:

1. **Settings Class**:
   - A singleton class for managing mutable configuration settings.
   - Supports type enforcement for settings to ensure correct usage.
   - Allows dynamic modification of settings using a context manager for temporary changes.
   - Retrieves environment variable overrides for settings, enabling external customization.

   #### Available Settings:
   - `allow_unverified_code` (bool, default: False): Whether to allow unverified code execution.
   - `use_only_local_catalogs` (bool, default: False): Restrict operations to local catalogs only.
   - `global_loader_limit` (int, default: None): Limit for global data loaders.
   - `num_resamples_for_instance_metrics` (int, default: 1000): Number of resamples for instance-level metrics.
   - `num_resamples_for_global_metrics` (int, default: 100): Number of resamples for global metrics.
   - `max_log_message_size` (int, default: 100000): Maximum size of log messages.
   - `catalogs` (default: None): List of catalog configurations.
   - `artifactories` (default: None): Artifact storage configurations.
   - `default_recipe` (str, default: "dataset_recipe"): Default recipe for dataset operations.
   - `default_verbosity` (str, default: "info"): Default verbosity level for logging.
   - `use_eager_execution` (bool, default: False): Enable eager execution for tasks.
   - `remote_metrics` (list, default: []): List of remote metrics configurations.
   - `test_card_disable` (bool, default: False): Disable test cards if set to True.
   - `test_metric_disable` (bool, default: False): Disable test metrics if set to True.
   - `metrics_master_key_token` (default: None): Master token for metrics.
   - `seed` (int, default: 42): Default seed for random operations.
   - `skip_artifacts_prepare_and_verify` (bool, default: False): Skip artifact preparation and verification.
   - `data_classification_policy` (default: None): Policy for data classification.
   - `mock_inference_mode` (bool, default: False): Enable mock inference mode.
   - `disable_hf_datasets_cache` (bool, default: True): Disable caching for Hugging Face datasets.
   - `loader_cache_size` (int, default: 1): Cache size for data loaders.
   - `task_data_as_text` (bool, default: True): Represent task data as text.
   - `default_provider` (str, default: "watsonx"): Default service provider.
   - `default_format` (default: None): Default format for data processing.

   #### Usage:
   - Access settings using `get_settings()` function.
   - Modify settings temporarily using the `context` method:
     ```python
     settings = get_settings()
     with settings.context(default_verbosity="debug"):
         # Code within this block uses "debug" verbosity.
     ```

2. **Constants Class**:
   - A singleton class for managing immutable constants used across the application.
   - Constants cannot be modified once set.
   - Provides centralized access to paths, URLs, and other fixed application parameters.

   #### Available Constants:
   - `dataset_file`: Path to the dataset file.
   - `metric_file`: Path to the metric file.
   - `local_catalog_path`: Path to the local catalog directory.
   - `package_dir`: Directory of the installed package.
   - `default_catalog_path`: Default catalog directory path.
   - `dataset_url`: URL for dataset resources.
   - `metric_url`: URL for metric resources.
   - `version`: Current version of the application.
   - `catalog_hierarchy_sep`: Separator for catalog hierarchy levels.
   - `env_local_catalogs_paths_sep`: Separator for local catalog paths in environment variables.
   - `non_registered_files`: List of files excluded from registration.
   - `codebase_url`: URL of the codebase repository.
   - `website_url`: Official website URL.
   - `inference_stream`: Name of the inference stream constant.
   - `instance_stream`: Name of the instance stream constant.
   - `image_tag`: Default image tag for operations.
   - `demos_pool_field`: Field name for demos pool.

   #### Usage:
   - Access constants using `get_constants()` function:
     ```python
     constants = get_constants()
     print(constants.dataset_file)
     ```

3. **Helper Functions**:
   - `get_settings()`: Returns the singleton `Settings` instance.
   - `get_constants()`: Returns the singleton `Constants` instance.
"""
import importlib.metadata
import importlib.util
import os
from contextlib import contextmanager

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

    @classmethod
    def is_uninitilized(cls):
        return cls._instance is None

    def __new__(cls):
        if cls.is_uninitilized():
            cls._instance = super().__new__(cls)
        return cls._instance

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

        if key in self._settings:
            if self._logger is not None:
                self._logger.info(
                    f"unitxt.settings.{key} changed: {self._settings[key]} -> {value}"
                )
        self._settings[key] = value

    def __getattr__(self, key):
        if key.endswith("_key"):
            actual_key = key[:-4]  # Remove the "_key" suffix
            return self.environment_variable_key_name(actual_key)

        key_name = self.environment_variable_key_name(key)
        env_value = os.getenv(key_name)

        if env_value is not None:
            if key in self._types:
                env_value = cast_to_type(env_value, self._types[key])
            return env_value

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
        old_values = {key: self._settings.get(key, None) for key in kwargs}
        try:
            for key, value in kwargs.items():
                self.__setattr__(key, value)
            yield
        finally:
            for key, value in old_values.items():
                self.__setattr__(key, value)


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
    settings.disable_hf_datasets_cache = (bool, True)
    settings.loader_cache_size = (int, 1)
    settings.task_data_as_text = (bool, True)
    settings.default_provider = "watsonx"
    settings.default_format = None

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


def get_settings() -> Settings:
    return Settings()


def get_constants():
    return Constants()
