import glob
import os
import sys
import time
from datetime import timedelta

from unitxt.logging_utils import get_logger
from unitxt.settings_utils import get_constants, get_settings
from unitxt.text_utils import print_dict
from unitxt.utils import import_module_from_file

from tests.utils import UnitxtExamplesTestCase

logger = get_logger()
constants = get_constants()
settings = get_settings()

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
glob_query = os.path.join(project_dir, "examples", "**", "*.py")
all_example_files = glob.glob(glob_query, recursive=True)

_excluded_files = [
    "evaluate_a_judge_model_capabilities_on_arena_hard.py",
    "evaluate_using_metrics_ensemble.py",
    "evaluate_existing_dataset_no_install.py",
    "evaluate_existing_dataset_by_llm_as_judge.py",
    "custom_type.py",
    "evaluate_batched_multiclass_classification.py",
    "evaluate_llm_as_judge.py",
    "evaluate_different_templates_num_demos.py",
    "evaluate_image_text_to_text_vllm_inference.py",
    "evaluate_qa_dataset_with_given_predictions.py",
    "evaluate_ensemble_judge.py",
    "evaluate_benchmark.py",
    "evaluate_classification_dataset_with_given_predictions.py",
    "evaluate_rag_end_to_end_dataset_with_given_predictions.py",
    "evaluate_grounded_ensemble_judge.py",
    "evaluate_rag_response_generation.py",
    # uncleared:
    "evaluate_idk_judge.py",
    "evaluate_existing_dataset_with_install.py",
    "evaluate_rag.py",
    "robustness_testing_for_vision_text_models.py",
    "evaluate_a_model_using_arena_hard.py",
    "evaluate_bluebench.py",
    "evaluate_image_text_to_text_lmms_eval_inference.py",
    "evaluate_different_demo_selections.py",
    # "inference_using_ibm_watsonx_ai.py",
    # "evaluate_image_text_to_text.py",
    # "qa_evaluation.py",
    # "evaluate_different_formats.py",
    # "evaluate_benchmark_with_custom_provider.py",
    # "standalone_evaluation_llm_as_judge.py",
    # "evaluate_summarization_dataset_llm_as_judge.py",
    # "evaluate_different_templates.py",
    # "evaluate_rag_using_binary_llm_as_judge.py",
    # "evaluate_image_text_to_text_with_different_templates.py",
    # "evaluate_external_rag_results_with_binary_llm_as_judge.py",
    # "evaluate_same_datasets_and_models_with_multiple_providers.py",
    # "standalone_qa_evaluation.py",
]


class TestExamples(UnitxtExamplesTestCase):
    def test_examples(self):
        logger.info(glob_query)

        tested_files = [
            file
            for file in all_example_files
            if os.path.basename(file) not in _excluded_files
        ]
        logger.critical(f"Testing example files: {tested_files}")
        # Make sure the order in which the tests are run is deterministic
        # Having a different order for local testing and github testing may cause diffs in results.
        times = {}
        tested_files.sort()
        failed_examples_files = []
        for file in tested_files:
            file_name = os.path.basename(file)
            logger.info(
                "\n_____________________________________________\n"
                f"  Testing examples file:\n  {file_name}."
                "\n_____________________________________________\n"
            )

            start_time = time.time()
            with self.subTest(file=file_name):
                try:
                    _original_stdout = sys.stdout
                    sys.stdout = open(os.devnull, "w")

                    import_module_from_file(file)

                    sys.stdout.close()
                    sys.stdout = _original_stdout

                    logger.info(f"Testing example file: {file_name} passed")
                    self.assertTrue(True)
                except Exception as e:
                    logger.error(
                        f"\nTesting example file: {file_name}\nFailed due to:\n{e!s}"
                    )
                    failed_examples_files.append(file)
                    self.assertTrue(False)
            elapsed_time = time.time() - start_time
            formatted_time = str(timedelta(seconds=elapsed_time))
            logger.info(
                "\n_____________________________________________\n"
                f"  Finished testing example file:\n  {file_name}\n"
                f"  Run Time: {formatted_time}"
                "\n_____________________________________________\n"
            )
            times[file] = formatted_time
        logger.critical("Example run time:")
        print_dict(times, log_level="critical")
        if len(failed_examples_files) > 0:
            logger.error("Failed examples:")
            logger.error(failed_examples_files)
        self.assertLessEqual(len(failed_examples_files), 0)
