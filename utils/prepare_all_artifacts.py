from glob import glob
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


def prepare_all(artifact_type):
    all_prepare_paths = [
        str(p) for p in Path(f"./prepare/{artifact_type}").glob(f"**/*.py")
    ]
    #all_prepare_paths = list(glob(os.path.join('./prepare', '**', '*.py'), recursive=True))
    errors = []
    for file in all_prepare_paths:
        print("*" * 100)
        print("* " + file)
        print("*")
        try:
            import_module_from_file(file)
        except Exception as e:
            errors.append({'file': file, 'exception': e})

    if errors:
        print(f'has {len(errors)} errors: ')
        for error in errors:


            print(error)


def main():
    settings = get_settings()
    settings.use_only_local_catalogs = True
   # os.environ["UNITXT_USE_ONLY_LOCAL_CATALOGS"] = 'True'
    os.environ["UNITXT_TEST_CARD_DISABLE"] = "True"
    os.environ["UNITXT_TEST_METRIC_DISABLE"] = "True"
    print("*" * 100)
    print("*" * 100)
    print("* DELETING OLD FM_EVAL CATALOG  *** ")
    print("deleting all files from 'src/unitxt/catalog'")
    shutil.rmtree('./src/unitxt/catalog', ignore_errors=True)
    prepare_all("tasks")
    prepare_all("processors")
    prepare_all("operators")
    prepare_all("templates")
    prepare_all("metrics")
    prepare_all("cards")
    prepare_all("formats")
    prepare_all("system_prompts")
    prepare_all("augmentors")


if __name__ == '__main__':
    main()