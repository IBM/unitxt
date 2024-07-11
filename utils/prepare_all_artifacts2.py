import importlib.util
import os

from unitxt import get_logger
from unitxt.settings_utils import get_constants, get_settings

logger = get_logger()
constants = get_constants()
settings = get_settings()


def import_module_from_file(file_path):
    # Get the module name (file name without extension)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    # Create a module specification
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    # Create a new module based on the specification
    module = importlib.util.module_from_spec(spec)
    # Load the module
    logger.info(
        f"allow unverified code in {file_path} : {settings.allow_unverified_code}"
    )
    spec.loader.exec_module(module)
    return module


def prepare_artifacts_for_prepare_files(prepare_files):
    failed_prepare_files = []
    prepare_exceptions = []
    for i, file in enumerate(prepare_files):
        logger.info("*" * 100)
        logger.info(f"* {i}/{len(prepare_files)}: {file}")
        logger.info("*")
        try:
            import_module_from_file(file)
        except Exception as e:
            logger.info(f"Failed to prepare: {file}")
            failed_prepare_files.append(file)
            prepare_exceptions.append(e)

    return failed_prepare_files, prepare_exceptions


def main():
    os.environ["UNITXT_USE_ONLY_LOCAL_CATALOGS"] = "True"
    os.environ["UNITXT_TEST_CARD_DISABLE"] = "True"
    os.environ["UNITXT_TEST_METRIC_DISABLE"] = "True"
    os.environ["UNITXT_SKIP_ARTIFACTS_PREPARE_AND_VERIFY"] = "True"
    logger.info("*" * 100)
    logger.info("*" * 100)
    logger.info("* DELETING OLD FM_EVAL CATALOG  *** ")
    logger.info("deleting all files from 'src/unitxt/catalog'")
    import_module_from_file(
        "/Users/eladv/unitxt/prepare/metrics/rag_context_correctness.py"
    )


if __name__ == "__main__":
    main()
