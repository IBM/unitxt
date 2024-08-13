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
        # unitxt.settings.global_loader_limit = 300
        unitxt.settings.max_log_message_size = 1000000000000
        if settings.default_verbosity in ["error", "critical"]:
            if not sys.warnoptions:
                warnings.simplefilter("ignore")
        register_local_catalog_for_tests()
        cls.maxDiff = None


def fillna(data, fill_value):
    import numpy as np

    if isinstance(data, dict):
        return {k: fillna(v, fill_value) for k, v in data.items()}
    if isinstance(data, list):
        return [fillna(item, fill_value) for item in data]
    return fill_value if isinstance(data, float) and np.isnan(data) else data
