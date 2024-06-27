import unittest

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


class UnitxtCatalogPreparationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enable_explicit_format()
        unitxt.settings.allow_unverified_code = True
        unitxt.settings.use_only_local_catalogs = True
        # unitxt.settings.global_loader_limit = 300
        unitxt.settings.max_log_message_size = 1000
        register_local_catalog_for_tests()
        cls.maxDiff = None
