import sys
import unittest
import warnings

import unitxt
from unitxt.logging_utils import enable_explicit_format, get_logger
from unitxt.settings_utils import get_settings
from unitxt.test_utils.catalog import register_local_catalog_for_tests

settings = get_settings()
logger = get_logger()


class UnitxtTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enable_explicit_format()
        unitxt.settings.allow_unverified_code = True
        unitxt.settings.use_only_local_catalogs = True
        unitxt.settings.global_loader_limit = 300
        unitxt.settings.max_log_message_size = 10000
        sys.tracebacklimit = None
        register_local_catalog_for_tests()
        cls.maxDiff = None
        if settings.default_verbosity in ["error", "critical"]:
            if not sys.warnoptions:
                warnings.simplefilter("ignore")


class UnitxtCatalogPreparationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enable_explicit_format()
        unitxt.settings.allow_unverified_code = True
        unitxt.settings.use_only_local_catalogs = True
        unitxt.settings.mock_inference_mode = True
        unitxt.settings.max_log_message_size = 1000000000000
        if settings.default_verbosity in ["error", "critical"]:
            if not sys.warnoptions:
                warnings.simplefilter("ignore")
        register_local_catalog_for_tests()
        cls.maxDiff = None


class UnitxtInferenceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enable_explicit_format()
        unitxt.settings.allow_unverified_code = True
        unitxt.settings.use_only_local_catalogs = True
        unitxt.settings.global_loader_limit = 300
        unitxt.settings.max_log_message_size = 10000
        sys.tracebacklimit = None
        register_local_catalog_for_tests()
        cls.maxDiff = None
        if settings.default_verbosity in ["error", "critical"]:
            if not sys.warnoptions:
                warnings.simplefilter("ignore")


def apply_recursive(data, func):
    if isinstance(data, dict):
        return {k: apply_recursive(v, func) for k, v in data.items()}
    if isinstance(data, list):
        return [apply_recursive(item, func) for item in data]
    return func(data)


def fillna(data, fill_value):
    import numpy as np

    def fill_func(x):
        return fill_value if isinstance(x, float) and np.isnan(x) else x

    return apply_recursive(data, fill_func)


def round_values(data, points=3):
    def round_func(x):
        if isinstance(x, (int, float)):
            return round(x, points)
        return x

    return apply_recursive(data, round_func)
