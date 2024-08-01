from pathlib import Path

from datasets.utils.py_utils import get_imports

from .file_utils import get_all_files_in_dir


def get_missing_imports(file, exclude=None):
    if exclude is None:
        exclude = []
    src_dir = Path(__file__).parent
    python_files = get_all_files_in_dir(src_dir, file_extension=".py")
    # get only the file without the path and extension
    required_modules = [Path(p).stem for p in python_files]
    imports = get_imports(file)
    imported_modules = [i[1] for i in imports if i[0] == "internal"]
    return [
        i for i in required_modules if i not in imported_modules and i not in exclude
    ]
