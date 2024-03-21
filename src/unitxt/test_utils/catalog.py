import os
import tempfile
from contextlib import contextmanager

from ..register import (
    register_local_catalog,
    unregister_local_catalog,
)
from ..settings_utils import get_constants, get_settings

constants = get_constants()
settings = get_settings()


def register_local_catalog_for_tests():
    os.environ[settings.artifactories_key] = constants.catalog_dir
    # _reset_env_local_catalogs()


@contextmanager
def temp_catalog():
    with tempfile.TemporaryDirectory() as temp_dir:
        register_local_catalog(temp_dir)
        try:
            yield temp_dir
        finally:
            unregister_local_catalog(temp_dir)
