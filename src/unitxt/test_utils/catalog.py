import os

from .. import __file__ as unitxt_file
from ..register import UNITXT_ARTIFACTORIES_ENV_VAR, _reset_env_local_catalogs


def register_local_catalog_for_tests():
    unitxt_dir = os.path.dirname(unitxt_file)
    catalog_dir = os.path.join(unitxt_dir, "catalog")
    os.environ[UNITXT_ARTIFACTORIES_ENV_VAR] = catalog_dir
    _reset_env_local_catalogs()
