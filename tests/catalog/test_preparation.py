import glob
import os
import time
import tracemalloc

import psutil
from huggingface_hub.utils import GatedRepoError
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
glob_query = os.path.join(project_dir, "prepare", "**", "*.py")
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

skip_files = [
    "/augmentors/table_augmentors.py",
    "/cards/20newsgroups_sklearn.py",
    "/cards/arena_hard/common.py",
    "/cards/attaq_500.py",
    "/cards/chat_rag_bench.py",
    "/cards/copa.py",
    "/cards/ffqa_filtered.py",
    "/cards/head_qa.py",
    "/cards/legalbench.py",
    "/cards/mnli.py",
    "/cards/mt_bench/response_assessment/pairwise_comparison/single_turn_with_reference_gpt4_judgement.py",
    # Checked
    "/cards/numeric_nlg.py",
    "/cards/qtsumm.py",
    "/cards/rte.py",
    "/cards/seed_bench.py",
    # "/cards/tablebench.py",
    # "/cards/translation/flores101.py",
    # "/cards/websrc.py",
    # "/cards/xsum.py",
    "/engines/model/flan.py",
    "/formats/human_assistant.py",
    "/formats/models/llama3.py",
    "/formats/user_assistant.py",
    "/metrics/exact_match_mm.py",
    "/metrics/llama_index_metrics.py",
    "/metrics/llm_as_judge/rating/llama_3_1_cross_provider_table2text_template.py",
    "/metrics/matthews_correlation.py",
    "/metrics/qa.py",
    "/metrics/rag_metrics_deprecated.py",
    "/metrics/safety_metric.py",
    "/metrics/wer.py",
    "/processors/to_list_by_hyphen.py",
    "/splitters/missing_split.py",
    "/system_prompts/models/llama2.py",
    "/tasks/language_identification.py",
    "/tasks/response_assessment/pairwise_comparative_rating/single_turn.py",
    "/tasks/response_assessment/rating/single_turn_with_reference.py",
    "/templates/classification/classification.py",
    "/templates/generation/generation.py",
    "/templates/rag_eval/rag_eval_numeric.py",
    "/templates/response_assessment/judges/topicality/v3.py",
    "/templates/response_assessment/rating/generic_single_turn_with_reference.py",
    "/templates/safety/harm_rating.py",
    "/templates/translation/directed.py",
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
            # if file.split("prepare")[-1] in skip_files:
            #     continue
            logger.info(
                "\n_____________________________________________\n"
                f"  Testing preparation file:\n  {file}."
                "\n_____________________________________________\n"
            )
            try:
                process = psutil.Process()
                start_memory = process.memory_info().rss / (
                    1024**3
                )  # Convert bytes to GB
                disk_start = psutil.disk_io_counters()
                start_time = time.time()
                tracemalloc.start()

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
            except Exception as e:
                logger.critical(f"Testing preparation file '{file}' failed:")
                raise e

        logger.critical(f"Preparation times table for {len(stats)} files:")
        times = dict(sorted(stats.items(), key=lambda item: item[1], reverse=True))
        print_dict(times, log_level="critical")
