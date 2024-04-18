from pathlib import Path

from datasets.utils.py_utils import get_imports

from .deprecation_utils import compare_versions
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


class UnitxtVersionsConflictError(ValueError):
    def __init__(self, error_in: str, hf_unitxt_version, installed_unitxt_version):
        assert hf_unitxt_version != installed_unitxt_version
        if compare_versions(hf_unitxt_version, installed_unitxt_version) == 1:
            msg = f"Located installed unitxt version {installed_unitxt_version} that is older than unitxt {error_in} version {hf_unitxt_version}. Please update unitxt package or uninstall it to avoid conflicts."
        if compare_versions(hf_unitxt_version, installed_unitxt_version) == -1:
            msg = f"Located installed unitxt version {installed_unitxt_version} that is newer than unitxt {error_in} version {hf_unitxt_version}. Please force-reload the {error_in} or downgrade unitxt to {error_in} version or uninstall unitxt to avoid conflicts."
        super().__init__(msg)


def get_installed_version():
    from unitxt.settings_utils import get_constants as installed_get_constants

    return installed_get_constants().version


def verify_versions_compatibility(hf_asset_type, hf_asset_version):
    _verify_versions(hf_asset_type, get_installed_version(), hf_asset_version)


def _verify_versions(hf_asset_type, installed_version, hf_asset_version):
    if installed_version != hf_asset_version:
        raise UnitxtVersionsConflictError(
            hf_asset_type, hf_asset_version, installed_version
        )
