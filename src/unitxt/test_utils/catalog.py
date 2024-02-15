import os

from ..register import UNITXT_ARTIFACTORIES_ENV_VAR, _reset_env_local_catalogs
from ..settings_utils import get_constants

constants = get_constants()


def register_local_catalog_for_tests():
    os.environ[UNITXT_ARTIFACTORIES_ENV_VAR] = constants.catalog_dir
    _reset_env_local_catalogs()
