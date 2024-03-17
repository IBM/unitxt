import filecmp
from pathlib import Path
import importlib.util
import os
import shutil
from src.unitxt.settings_utils import get_settings


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


def prepare_all_artifacts_in_catalog_for_type(artifact_type):
    all_prepare_paths = [
        str(p) for p in Path(f"./prepare/{artifact_type}").glob(f"**/*.py")
    ]
    errors = []
    for file in all_prepare_paths:
        print("*" * 100)
        print("* " + file)
        print("*")
        try:
            import_module_from_file(file)
        except Exception as e:
            errors.append({'file': file, 'exception': e})

    return errors


def prepare_all_catalog_artifacts():
    settings = get_settings()
    settings.use_only_local_catalogs = True
    os.environ["UNITXT_TEST_CARD_DISABLE"] = "True"
    os.environ["UNITXT_TEST_METRIC_DISABLE"] = "True"
    print("*" * 100)
    print("*" * 100)
    print("* DELETING OLD FM_EVAL CATALOG  *** ")
    print("deleting all files from 'src/unitxt/catalog'")
    shutil.rmtree('./src/unitxt/catalog', ignore_errors=True)
    catalog_types = ["tasks", "processors", "operators", "templates", "metrics", "splitters", "cards", "formats",
                     "system_prompts", "augmentors"]
    errors = []
    for type in catalog_types:
        errors.extend(prepare_all_artifacts_in_catalog_for_type(type))

    if errors:
        print(f'has {len(errors)} errors: ')
        for error in errors:
            print(error)


def compare_dirs(old, new):
    dir_cmp = filecmp.dircmp(old, new)
    diffs = []
    if dir_cmp.diff_files or dir_cmp.left_only or dir_cmp.right_only or dir_cmp.funny_files:
        if dir_cmp.left_only:
            diffs.extend([{'file': os.path.join(new, file), 'diff': 'old only'} for file in dir_cmp.left_only])
        if dir_cmp.right_only:
            diffs.extend([{'file': os.path.join(new, file), 'diff': 'new only'} for file in dir_cmp.right_only])
        if dir_cmp.diff_files:
            diffs.extend([{'file': os.path.join(new, file), 'diff': 'diff'} for file in dir_cmp.diff_files])
        if dir_cmp.funny_files:
            diffs.extend([{'file': os.path.join(new, file), 'diff': 'failed'} for file in dir_cmp.funny_files])

    # Recursively compare subdirectories
    for sub_dir, sub_dcmp in dir_cmp.subdirs.items():
        diffs.extend(compare_dirs(os.path.join(old, sub_dir), os.path.join(new, sub_dir)))

    return diffs


def filter_known_diffs(diffs):
    return [diff for diff in diffs if
            'news_category_classification_headline' not in diff['file'] and  # in order to create we need Kaggle credentials
            'tablerow_classify' not in diff['file']]                        # in order to create we need Kaggle credentials


def main():
    old_dir = './src/unitxt/catalog_old'
    new_dir = './src/unitxt/catalog'
    print('move old catalog:')
    try:
        shutil.rmtree(old_dir)
    except:
        pass
    shutil.move(new_dir, old_dir)
    print('Starting reprepare catalog...')
    prepare_all_catalog_artifacts()
    print('Comparing generated and old catalog...')
    diffs = compare_dirs(new=new_dir, old=old_dir)
    diffs = filter_known_diffs(diffs)
    if diffs:
        print('***** Directories has differences ******')
        diffs.sort(key=lambda d: d['file'])
        for diff in diffs:
            print(diff)
        raise RuntimeError('Directories has differences')


if __name__ == '__main__':
    main()