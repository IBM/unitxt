import glob
import importlib.util
import os
import shutil

from unitxt import get_logger
from unitxt.settings_utils import get_constants, get_settings

logger = get_logger()
constants = get_constants()
settings = get_settings()

# put here the absolute path to the dir containing all prepare files - potentially, partitioned into subdirs"
prepare_dir = "/home/user/workspaces/unitxt/prepare"

# put here the absolute path to the dir where the catalog is to be generated into."
catalog_dir = "/home/user/workspaces/unitxt/src/unitxt/catalog2"
#
# Note: set the following constant in settings_utils.py:
# constants.default_catalog_path = catalog_dir
#


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


# flake8: noqa: C901
def main():
    # create a clean catalog_dir
    shutil.rmtree(catalog_dir, ignore_errors=True)
    os.makedirs(catalog_dir, exist_ok=True)

    os.environ["UNITXT_USE_ONLY_LOCAL_CATALOGS"] = "True"
    os.environ["UNITXT_TEST_CARD_DISABLE"] = "True"
    os.environ["UNITXT_TEST_METRIC_DISABLE"] = "True"
    os.environ["UNITXT_ALLOW_UNVERIFIED_CODE"] = "True"
    os.environ["UNITXT_SKIP_ARTIFACTS_PREPARE_AND_VERIFY"] = "True"
    logger.info("*" * 100)
    logger.info("*" * 100)

    logger.critical("Starting to reprepare the catalog...")
    prepare_files = sorted(glob.glob(f"{prepare_dir}/**/*.py", recursive=True))
    # prepare_files = ["/home/dafna/workspaces/unitxt/prepare/cards/coqa.py"]
    failing_prepare_files = []
    rounds = 0
    while True:
        initial_number_of_catalog_entries = len(
            glob.glob(f"{catalog_dir}/**/*.json", recursive=True)
        )
        rounds += 1
        logger.info("*" * 100)
        logger.info("*" * 100)
        logger.info(f"******************** round {rounds} ********")
        logger.info("*" * 100)
        logger.info("*" * 100)

        for i, prepare_file in enumerate(prepare_files):
            logger.info("*" * 100)
            logger.info(f"* {i+1}/{len(prepare_files)}: {prepare_file}")
            logger.info("*")
            try:
                import_module_from_file(prepare_file)

            except Exception as e:
                logger.info(
                    f"Failed to generate at least one catalog entry by prepare file: {prepare_file} for reason {e}"
                )
                failing_prepare_files.append(prepare_file)
        if len(failing_prepare_files) == 0:
            break
        final_number_of_catalog_entries = len(
            glob.glob(f"{catalog_dir}/**/*.json", recursive=True)
        )
        if final_number_of_catalog_entries <= initial_number_of_catalog_entries:
            error_msg = f"all the following {len(prepare_files)} prepare files fail forever: {prepare_files}. "
            "One potential reason is a circular dependency among them, another is that at least one of them contains add_link_to_catalog "
            "of an ArtifactLink that links to an artifact that is added to the catalog only down that prepare_file. "
            "To fix: resolve dependency, or swap the order: first add_to_catalog the artifact linked to, and then add_link_to_catalog."
            raise RuntimeError(error_msg)
        prepare_files = failing_prepare_files
        failing_prepare_files = []

    final_number_of_catalog_entries = len(
        glob.glob(f"{catalog_dir}/**/*.json", recursive=True)
    )
    logger.info(
        f"Completed to generate all {final_number_of_catalog_entries} catalog entries, by running all prepare files."
    )


if __name__ == "__main__":
    main()
