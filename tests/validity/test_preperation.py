import glob
import importlib.util
import os
import unittest

from src.unitxt.loaders import MissingKaggleCredentialsError
from src.unitxt.logging_utils import get_logger

os.environ["UNITXT_TEST_CARD_DISABLE"] = "True"
os.environ["UNITXT_TEST_METRIC_DISABLE"] = "True"

logger = get_logger()
project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
prepration_dir = os.path.join(project_dir, "prepare")
catalog_dir = os.path.join(project_dir, "src", "unitxt", "catalog")


def test_all_python_modules_in_dir(dir, tester):
    glob_query = os.path.join(dir, "**", "*.py")
    all_preparation_files = glob.glob(glob_query, recursive=True)

    logger.info(glob_query)
    logger.info(f"Testing preparation files: {all_preparation_files}")
    # Make sure the order in which the tests are run is deterministic
    # Having a different order for local testing and github testing may cause diffs in results.
    all_preparation_files.sort()
    for file in all_preparation_files:
        with tester.subTest(file=file):
            logger.info(f"Testing preparation file: {file}.")
            try:
                import_module_from_file(file)
            except MissingKaggleCredentialsError as e:
                logger.info(f"Skipping file {file} due to ignored error {e}")
                continue
            logger.info(f"Testing preparation file: {file} passed")
            tester.assertTrue(True)


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
    def test_prepration(self):
        dirs = [
            "metrics",
            "processors",
            "splitters",
            "instructions",
            "augmentors",
            "tasks",
            "templates",
            "formats",
            "cards",
        ]
        for dir in dirs:
            test_all_python_modules_in_dir(os.path.join(prepration_dir, dir), self)
