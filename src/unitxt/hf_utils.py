import re
from pathlib import Path
from typing import List

from .deprecation_utils import compare_versions
from .file_utils import get_all_files_in_dir


def get_internal_imports(file_path: str) -> List[str]:
    """Return a list of local (relative) modules directly imported in the given Python file."""
    internal_imports = []
    is_in_docstring = False
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.count('"""') == 1 or line.count("'''") == 1:
                is_in_docstring = not is_in_docstring
            if is_in_docstring:
                continue
            # Match "import .module" or "from .module import ..."
            match = re.match(r"^(?:import|from)\s+\.(\w+)", line)
            if match:
                module = match.group(1)
                if module not in internal_imports:
                    internal_imports.append(module)
    return internal_imports


def get_missing_imports(file, exclude=None):
    if exclude is None:
        exclude = []
    src_dir = Path(__file__).parent
    python_files = get_all_files_in_dir(src_dir, file_extension=".py")
    # get only the file without the path and extension
    required_modules = [Path(p).stem for p in python_files]
    imported_modules = get_internal_imports(file)
    return [
        i for i in required_modules if i not in imported_modules and i not in exclude
    ]


class UnitxtVersionsConflictError(ValueError):
    def __init__(self, error_in: str, hf_unitxt_version, installed_unitxt_version):
        assert hf_unitxt_version != installed_unitxt_version
        if compare_versions(hf_unitxt_version, installed_unitxt_version) == 1:
            msg = f"Located locally installed Unitxt version {installed_unitxt_version} that is older than the Huggingface Unitxt {error_in} version {hf_unitxt_version}. Please either (1) update the local Unitxt package or (2) uninstall the local unitxt package (3) remove the calls to the  Huggingface {error_in} API and use only the direct Unitxt APIs."
        if compare_versions(hf_unitxt_version, installed_unitxt_version) == -1:
            msg = f"Located locally installed Unitxt version {installed_unitxt_version} that is newer than the Huggingface Unitxt {error_in} version {hf_unitxt_version}. Please either (1) force-reload the {error_in} version or (2) downgrade the locally installed Unitxt version to {error_in} version or (3) uninstall the locally installed Unitxt, if you are not using the direct Unitxt APIs"
        msg += "For more details see: https://unitxt.readthedocs.io/en/latest/docs/installation.html"
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
