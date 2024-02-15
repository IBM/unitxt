import os

from ..register import _reset_env_local_catalogs
from ..settings_utils import get_constants, get_settings

constants = get_constants()
settings = get_settings()


def register_local_catalog_for_tests():
    os.environ[settings.artifactories_key] = constants.catalog_dir
    _reset_env_local_catalogs()
