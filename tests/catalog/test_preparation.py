import glob
import importlib.util
import os
import time
from datetime import timedelta

from huggingface_hub.utils import GatedRepoError
from unitxt.loaders import MissingKaggleCredentialsError
from unitxt.logging_utils import get_logger
from unitxt.settings_utils import get_constants
from unitxt.text_utils import print_dict

from tests.utils import UnitxtCatalogPreparationTestCase

logger = get_logger()
constants = get_constants()


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


project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
glob_query = os.path.join(project_dir, "prepare", "**", "*.py")
all_preparation_files = glob.glob(glob_query, recursive=True)


class TestCatalogPreparation(UnitxtCatalogPreparationTestCase):
    def test_preparations(self):
        logger.info(glob_query)
        logger.info(f"Testing preparation files: {all_preparation_files}")
        # Make sure the order in which the tests are run is deterministic
        # Having a different order for local testing and github testing may cause diffs in results.
        times = {}
        all_preparation_files.sort()
        for file in all_preparation_files:
            logger.info(
                "\n_____________________________________________\n"
                f"  Testing preparation file:\n  {file}."
                "\n_____________________________________________\n"
            )
            start_time = time.time()
            with self.subTest(file=file):
                try:
                    import_module_from_file(file)
                except (MissingKaggleCredentialsError, GatedRepoError) as e:
                    logger.info(f"Skipping file {file} due to ignored error {e}")
                    continue
                except OSError as e:
                    if "You are trying to access a gated repo" in str(e):
                        logger.info(f"Skipping file {file} due to ignored error {e}")
                        continue
                    raise
                logger.info(f"Testing preparation file: {file} passed")
                self.assertTrue(True)

            elapsed_time = time.time() - start_time
            formatted_time = str(timedelta(seconds=elapsed_time))
            logger.info(
                "\n_____________________________________________\n"
                f"  Finished testing preparation file:\n  {file}."
                f"  Preparation Time: {formatted_time}"
                "\n_____________________________________________\n"
            )

            times[file] = formatted_time
        logger.info("Preparation times table:")
        print_dict(times)
