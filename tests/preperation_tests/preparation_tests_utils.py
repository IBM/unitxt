# preparation_tests_utils.py

import glob
import importlib.util
import os
import unittest

from src.unitxt.logging import get_logger
from src.unitxt.random_utils import get_seed
from src.unitxt.test_utils.catalog import register_local_catalog_for_tests

logger = get_logger()
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
glob_query = os.path.join(project_dir, "prepare", "**", "*.py")
all_preparation_files = glob.glob(glob_query, recursive=True)


def import_module_from_file(file_path):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreparationTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_local_catalog_for_tests()

    @staticmethod
    def run_preparation_test(file):
        logger.info(f"Testing preparation file: {file}, current seed: {get_seed()}.")
        import_module_from_file(file)
        logger.info(f"Testing preparation file: {file} passed")


def get_preparation_test(shard, total):
    files_to_test = [
        f for i, f in enumerate(all_preparation_files) if i % total == shard - 1
    ]
    files_to_test.sort()

    class TestPreparation(PreparationTestBase):
        def test_preparations(self):
            logger.info(f"Testing preparation files for shard {shard} of {total}")
            for file in files_to_test:
                with self.subTest(file=file):
                    self.run_preparation_test(file)

    return TestPreparation
