import unittest

from src import unitxt
from src.unitxt.logging_utils import enable_explicit_format, get_logger
from src.unitxt.settings_utils import get_settings
from src.unitxt.test_utils.catalog import register_local_catalog_for_tests

settings = get_settings()
logger = get_logger()


class UnitxtTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enable_explicit_format()
        unitxt.settings.allow_unverified_code = True
        unitxt.settings.use_only_local_catalogs = True
        unitxt.settings.global_loader_limit = 300
        unitxt.settings.max_log_message_size = 1000
        register_local_catalog_for_tests()

    def setUp(self):
        logger.info(f"\n###  Running {self.__class__.__name__} ###\n")


class UnitxtCatalogPreparationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enable_explicit_format()
        unitxt.settings.allow_unverified_code = True
        unitxt.settings.use_only_local_catalogs = True
        unitxt.settings.global_loader_limit = 300
        register_local_catalog_for_tests()

    def setUp(self):
        logger.info(f"\n###  Running {self.__class__.__name__} ###\n")
