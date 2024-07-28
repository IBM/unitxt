import glob
import os
import time
from datetime import timedelta
from pathlib import Path

from unitxt.logging_utils import get_logger
from unitxt.settings_utils import get_constants
from unitxt.text_utils import print_dict
from unitxt.utils import import_module_from_file

from tests.utils import UnitxtTestCase

logger = get_logger()
constants = get_constants()


project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
glob_query = os.path.join(project_dir, "examples", "**", "*.py")
all_example_files = glob.glob(glob_query, recursive=True)


class TestExamples(UnitxtTestCase):
    def test_examples(self):
        logger.info(glob_query)
        logger.info(f"Testing example files: {all_example_files}")
        # Make sure the order in which the tests are run is deterministic
        # Having a different order for local testing and github testing may cause diffs in results.
        times = {}
        all_example_files.sort()

        excluded_files = [
            "use_llm_as_judge_metric.py",
            "standalone_evaluation_llm_as_judge.py",
            "evaluate_summarization_dataset_llm_as_judge.py",
            "evaluate_different_formats.py",
            "evaluate_different_templates.py",
            "evaluate_different_demo_selections.py",
            "evaluate_dataset_by_llm_as_judge_no_install.py",
            "evaluate_a_judge_model_capabilities_on_arena_hard.py",
            "evaluate_a_model_using_arena_hard.py",
            "evaluate_llm_as_judge.py",
            "evaluate_using_metrics_ensemble.py",
        ]
        for file in all_example_files:
            logger.info(
                "\n_____________________________________________\n"
                f"  Testing examples file:\n  {file}."
                "\n_____________________________________________\n"
            )
            if Path(file).name in excluded_files:
                logger.info("Skipping file because in exclude list")
                continue

            start_time = time.time()
            with self.subTest(file=file):
                import_module_from_file(file)
                logger.info(f"Testing example file: {file} passed")

            elapsed_time = time.time() - start_time
            formatted_time = str(timedelta(seconds=elapsed_time))
            logger.info(
                "\n_____________________________________________\n"
                f"  Finished testing examplefile:\n  {file}."
                f"  Preparation Time: {formatted_time}"
                "\n_____________________________________________\n"
            )

            times[file] = formatted_time
        logger.info("Example table:")
        print_dict(times)
