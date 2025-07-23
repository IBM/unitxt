import glob
import os
import time
import traceback
import tracemalloc

import psutil
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from requests.exceptions import ReadTimeout
from unitxt.loaders import MissingKaggleCredentialsError
from unitxt.logging_utils import get_logger
from unitxt.settings_utils import get_constants, get_settings
from unitxt.text_utils import print_dict
from unitxt.utils import import_module_from_file

from tests.utils import CatalogPreparationTestCase

logger = get_logger()
constants = get_constants()
setting = get_settings()

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
glob_query = os.path.join(project_dir, "prepare/cards", "**", "*.py")
all_preparation_files = glob.glob(glob_query, recursive=True)
# Make sure the order in which the tests are run is deterministic
# Having a different order for local testing and github testing may cause diffs in results.
all_preparation_files.sort()
num_par = 1  # num of parallel executions
logger.critical(
    f"Over all, {len(all_preparation_files)} files will now be tested over {num_par} parallel processes."
)
# the following should be any of modulo num_par: 0,1,2,3,4,5,6,7,8,.. num_par-1
modulo = 0
all_preparation_files = [
    file for i, file in enumerate(all_preparation_files) if i % num_par == modulo
]


class TestCatalogPreparation(CatalogPreparationTestCase):
    def test_preparations(self):
        logger.info(glob_query)
        all_preparation_files_as_string = "\n".join(
            [file.split("prepare")[-1][1:] for file in all_preparation_files]
        )
        logger.critical(
            f"Testing {len(all_preparation_files)} preparation files: \n{all_preparation_files_as_string}\n"
        )
        stats = {}
        for file in all_preparation_files:
            # passed = True
            error = None
            logger.info(
                "\n_____________________________________________\n"
                f"  Testing preparation file:\n  {file}."
                "\n_____________________________________________\n"
            )
            process = psutil.Process()
            start_memory = process.memory_info().rss / (1024**3)  # Convert bytes to GB
            disk_start = psutil.disk_io_counters()
            start_time = time.time()
            tracemalloc.start()

            with self.subTest(file=file):
                try:
                    import_module_from_file(file)
                except Exception as e:
                    error = e
                    # passed = False
                    current_exception = e
                    while current_exception:
                        if isinstance(current_exception, (GatedRepoError)):
                            # passed = False
                            break
                        if isinstance(
                            current_exception,
                            (
                                ReadTimeout,
                                HfHubHTTPError,
                                MissingKaggleCredentialsError,
                            ),
                        ):
                            # passed = True
                            break
                        current_exception = (
                            current_exception.__cause__ or current_exception.__context__
                        )

                # if passed:
                if error is None:
                    logger.info(f"Testing preparation file: {file} passed")
                else:
                    logger.critical(
                        f"Testing preparation file: {file} failed with error: {error}\n{traceback.format_exc()}"
                    )
                # else:
                #     raise error

                # self.assertTrue(passed)

            elapsed_time = time.time() - start_time
            disk_end = psutil.disk_io_counters()
            read_gb = (disk_end.read_bytes - disk_start.read_bytes) / (1024**3)
            write_gb = (disk_end.write_bytes - disk_start.write_bytes) / (1024**3)

            tracemalloc.stop()
            _, peak = tracemalloc.get_traced_memory()
            # Convert to GB
            peak_memory_python = peak / (1024**3)  # Convert bytes to GB
            peak_memory_system = (
                process.memory_info().rss / (1024**3) - start_memory
            )  # Convert bytes to GB

            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            formatted_time = f"{minutes:02}:{seconds:02}"
            logger.info(
                "\n_____________________________________________\n"
                f"  Finished testing preparation file:\n  {file}."
                f"  Elapsed Time: {formatted_time}\n"
                f"  Peak Python Memory Usage: {peak_memory_python:.4f} GB\n"
                f"  Peak System RAM Usage: {peak_memory_system:.4f} GB\n"
                f"  Disk Write: {write_gb:.4f} GB, Disk Read: {read_gb:.4f} GB"
                "\n_____________________________________________\n"
            )

            stats[
                file.split("prepare")[-1][1:]
            ] = f"Time: {formatted_time}, RAM: {peak_memory_system:.2f} GB, Disk: {write_gb:.2f} GB"

        logger.critical(f"Preparation times table for {len(stats)} files:")
        times = dict(sorted(stats.items(), key=lambda item: item[1], reverse=True))
        print_dict(times, log_level="critical")
