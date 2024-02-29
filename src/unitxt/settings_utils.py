import os

import pkg_resources

from .version import version


class Settings:
    _instance = None
    _settings = {}
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

        env_value = os.getenv(self.environment_variable_key_name(key))

        if env_value is not None:
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
    settings.allow_unverified_code = False
    settings.use_only_local_catalogs = False
    settings.global_loader_limit = None
    settings.num_resamples_for_instance_metrics = 1000
    settings.num_resamples_for_global_metrics = 100
    settings.max_log_message_size = 100000
    settings.artifactories = None
    settings.default_recipe = "standard_recipe"
    settings.default_verbosity = "debug"
    settings.remote_metrics = []

if Constants.is_uninitilized():
    constants = Constants()
    constants.dataset_file = os.path.join(os.path.dirname(__file__), "dataset.py")
    constants.metric_file = os.path.join(os.path.dirname(__file__), "metric.py")
    constants.local_catalog_path = os.path.join(os.path.dirname(__file__), "catalog")
    try:
        constants.default_catalog_path = pkg_resources.resource_filename(
            "unitxt", "catalog"
        )
    except ModuleNotFoundError:
        constants.default_catalog_path = constants.local_catalog_path
    constants.catalog_dir = constants.local_catalog_path
    constants.dataset_url = "unitxt/data"
    constants.metric_url = "unitxt/metric"
    constants.version = version
    constants.catalog_hirarchy_sep = "."
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


def get_settings():
    return Settings()


def get_constants():
    return Constants()
