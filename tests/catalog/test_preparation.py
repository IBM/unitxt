import glob
import os
import time

from huggingface_hub.utils import GatedRepoError
from unitxt.loaders import MissingKaggleCredentialsError
from unitxt.logging_utils import get_logger
from unitxt.settings_utils import get_constants, get_settings
from unitxt.text_utils import print_dict
from unitxt.utils import import_module_from_file

from tests.utils import UnitxtCatalogPreparationTestCase

logger = get_logger()
constants = get_constants()
setting = get_settings()

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
glob_query = os.path.join(project_dir, "prepare", "**", "*.py")
all_preparation_files = glob.glob(glob_query, recursive=True)
# Make sure the order in which the tests are run is deterministic
# Having a different order for local testing and github testing may cause diffs in results.
all_preparation_files.sort()
num_par = 1  # num of parallel executions
logger.critical(
    f"Over all, {len(all_preparation_files)} files will now be tested over {num_par} parallel processes."
)
# the following should be any of modulo num_par: 0,1,2,3,4,5,6,7,8,9,10,11
modulo = 0
if num_par > 1:
    all_preparation_files = [
        file for i, file in enumerate(all_preparation_files) if i % num_par == modulo
    ]


class TestCatalogPreparation(UnitxtCatalogPreparationTestCase):
    def test_preparations(self):
        logger.info(glob_query)
        logger.critical(
            f"Testing {len(all_preparation_files)} preparation files: {all_preparation_files}."
        )
        times = {}
        for file in all_preparation_files:
            logger.info(
                "\n_____________________________________________\n"
                f"  Testing preparation file:\n  {file}."
                "\n_____________________________________________\n"
            )
            try:
                start_time = time.time()
                with self.subTest(file=file):
                    try:
                        import_module_from_file(file)
                    except (MissingKaggleCredentialsError, GatedRepoError) as e:
                        logger.info(f"Skipping file {file} due to ignored error {e}")
                        continue
                    except OSError as e:
                        if "You are trying to access a gated repo" in str(e):
                            logger.info(
                                f"Skipping file {file} due to ignored error {e}"
                            )
                            continue
                        self.assertTrue(False)
                        raise
                    logger.info(f"Testing preparation file: {file} passed")
                    self.assertTrue(True)

                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                formatted_time = f"{minutes:02}:{seconds:02}"
                logger.info(
                    "\n_____________________________________________\n"
                    f"  Finished testing preparation file:\n  {file}."
                    f"  Preparation Time: {formatted_time}"
                    "\n_____________________________________________\n"
                )

                times[file.split("prepare")[-1]] = formatted_time
            except Exception as e:
                logger.critical(f"Testing preparation file '{file}' failed:")
                raise e

        logger.critical(f"Preparation times table for {len(times)} files:")
        times = dict(sorted(times.items(), key=lambda item: item[1], reverse=True))
        print_dict(times, log_level="critical")
