import glob
import importlib.util
import os
import sys
import unittest

from src.unitxt.random_utils import get_seed, set_seed
from src.unitxt.test_utils.catalog import register_local_catalog_for_tests

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
glob_query = os.path.join(project_dir, "prepare", "**", "*.py")
all_preparation_files = glob.glob(glob_query, recursive=True)


def import_module_from_file(file_path):
    # Get the module name (file name without extension)
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create a module specification
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    # Create a new module based on the specification
    module = importlib.util.module_from_spec(spec)

    # Load the module
    spec.loader.exec_module(module)

    return module


class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_local_catalog_for_tests()

    def test_preprations(self):
        print(glob_query)
        print(f"Testing preparation files: {all_preparation_files}")
        # Make sure the order in which the tests are run is deterministic
        # Having a different order for local testing and github testing may cause diffs in results.
        all_preparation_files.sort()
        for file in all_preparation_files:
            with self.subTest(file=file):
                print(f"Testing preparation file: {file}, current seed: {get_seed()}.")
                # Fix the random seed before loading the module. This is because for metrics,
                # a random generator member is used in confidence interval estimation. To make that
                # estimated confidence interval deterministic, the seed used for initializing the
                # random generator has to be fixed.
                set_seed(17)
                print(f"Seed was set to: {get_seed()}.")
                import_module_from_file(file)
                # with open(file, "r") as f:
                #     exec(f.read())
                print(f"Testing preparation file: {file} passed")
                self.assertTrue(True)
