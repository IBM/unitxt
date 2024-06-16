import filecmp
import glob
import importlib.util
import os
import shutil
from pathlib import Path

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


def prepare_all_catalog_artifacts(catalog_dir):
    os.environ["UNITXT_USE_ONLY_LOCAL_CATALOGS"] = "True"
    os.environ["UNITXT_TEST_CARD_DISABLE"] = "True"
    os.environ["UNITXT_TEST_METRIC_DISABLE"] = "True"
    os.environ["UNITXT_SKIP_ARTIFACTS_PREPARE_AND_VERIFY"] = "True"
    logger.info("*" * 100)
    logger.info("*" * 100)
    logger.info("* DELETING OLD FM_EVAL CATALOG  *** ")
    logger.info("deleting all files from 'src/unitxt/catalog'")
    shutil.rmtree(catalog_dir, ignore_errors=True)
    prepare_dir = os.path.join(Path(catalog_dir).parent.parent.parent, "prepare")
    prepare_files = sorted(glob.glob(f"{prepare_dir}/**/*.py", recursive=True))
    continue_preparing = True
    iteration = 0
    while continue_preparing:
        iteration += 1
        amount_of_prepare_files_before_iteration = len(prepare_files)
        logger.info(
            f"Iteration {iteration}: Preparing {amount_of_prepare_files_before_iteration} files"
        )
        prepare_files, prepare_exceptions = prepare_artifacts_for_prepare_files(
            prepare_files
        )
        if (
            len(prepare_files) == 0
            or len(prepare_files) == amount_of_prepare_files_before_iteration
            or iteration > 100
        ):
            continue_preparing = False
            logger.info(
                f"Done preparing files. Failed to prepare {len(prepare_files)} files:"
            )
            for file, exception in zip(prepare_files, prepare_exceptions):
                logger.info(f"Failed to prepare {file}. Exception: {exception}")


def compare_dirs(old, new):
    dir_cmp = filecmp.dircmp(old, new)
    diffs = []
    if (
        dir_cmp.diff_files
        or dir_cmp.left_only
        or dir_cmp.right_only
        or dir_cmp.funny_files
    ):
        if dir_cmp.left_only:
            diffs.extend(
                [
                    {"file": os.path.join(new, file), "diff": "old only"}
                    for file in dir_cmp.left_only
                ]
            )
        if dir_cmp.right_only:
            diffs.extend(
                [
                    {"file": os.path.join(new, file), "diff": "new only"}
                    for file in dir_cmp.right_only
                ]
            )
        if dir_cmp.diff_files:
            diffs.extend(
                [
                    {"file": os.path.join(new, file), "diff": "diff"}
                    for file in dir_cmp.diff_files
                ]
            )
        if dir_cmp.funny_files:
            diffs.extend(
                [
                    {"file": os.path.join(new, file), "diff": "failed"}
                    for file in dir_cmp.funny_files
                ]
            )

    # Recursively compare subdirectories
    for sub_dir, _ in dir_cmp.subdirs.items():
        diffs.extend(
            compare_dirs(os.path.join(old, sub_dir), os.path.join(new, sub_dir))
        )

    return diffs


def filter_known_diffs(diffs):
    return [
        diff
        for diff in diffs
        if "news_category_classification_headline"
        not in diff["file"]  # in order to create we need Kaggle credentials
        and "tablerow_classify" not in diff["file"]
    ]  # in order to create we need Kaggle credentials


def main():
    catalog_dir = constants.catalog_dir
    catalog_back_dir = catalog_dir + "_back"
    logger.info("move old catalog:")
    try:
        shutil.rmtree(catalog_back_dir)
    except:
        pass
    shutil.move(catalog_dir, catalog_back_dir)
    logger.info("Starting reprepare catalog...")
    prepare_all_catalog_artifacts(catalog_dir)
    logger.info("Comparing generated and old catalog...")
    diffs = compare_dirs(new=catalog_dir, old=catalog_back_dir)
    diffs = filter_known_diffs(diffs)
    if diffs:
        logger.info("***** Directories has differences ******")
        diffs.sort(key=lambda d: d["file"])
        for diff in diffs:
            logger.info(diff)
        raise RuntimeError("Directories has differences")
    logger.info("Done. Catalog is consistent with prepare files")


if __name__ == "__main__":
    main()
