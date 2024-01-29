import glob
import importlib.util
import os
import shutil
import tempfile
import unittest

from src.unitxt import register_local_catalog
from src.unitxt.loaders import MissingKaggleCredentialsError
from src.unitxt.logging_utils import get_logger
from src.unitxt.test_utils.catalog import register_local_catalog_for_tests

os.environ["UNITXT_TEST_CARD_DISABLE"] = "True"
os.environ["UNITXT_TEST_METRIC_DISABLE"] = "True"

logger = get_logger()
project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
catalog_dir = os.path.join(project_dir, "src", "unitxt", "catalog")
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


def move_all(src_dir, dst_dir):
    # Create the destination directory if it does not exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)

        # Move each item in the source directory to the destination directory
        shutil.move(src_path, dst_path)


class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_local_catalog_for_tests()

    def test_preprations(self):
        with tempfile.TemporaryDirectory() as tmp_directory:
            move_all(catalog_dir, tmp_directory)

            register_local_catalog(tmp_directory)
            register_local_catalog(catalog_dir)

            logger.info(glob_query)
            logger.info(f"Testing preparation files: {all_preparation_files}")
            # Make sure the order in which the tests are run is deterministic
            # Having a different order for local testing and github testing may cause diffs in results.
            all_preparation_files.sort()
            for file in all_preparation_files:
                with self.subTest(file=file):
                    logger.info(f"Testing preparation file: {file}.")
                    try:
                        import_module_from_file(file)
                    except MissingKaggleCredentialsError as e:
                        logger.info(f"Skipping file {file} due to ignored error {e}")
                        continue
                    logger.info(f"Testing preparation file: {file} passed")
                    self.assertTrue(True)
