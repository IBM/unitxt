import importlib
import inspect
import os

from .artifact import Artifact, Artifactories
from .catalog import PATHS_SEP, EnvironmentLocalCatalog, GithubCatalog, LocalCatalog
from .utils import Singleton

UNITXT_ARTIFACTORIES_ENV_VAR = "UNITXT_ARTIFACTORIES"

# Usage
non_registered_files = [
    "__init__.py",
    "artifact.py",
    "utils.py",
    "register.py",
    "metric.py",
    "dataset.py",
    "blocks.py",
]


def _register_catalog(catalog: LocalCatalog):
    Artifactories().register(catalog)


def _unregister_catalog(catalog: LocalCatalog):
    Artifactories().unregister(catalog)


def register_local_catalog(catalog_path: str):
    assert os.path.exists(catalog_path), f"Catalog path {catalog_path} does not exist."
    assert os.path.isdir(
        catalog_path
    ), f"Catalog path {catalog_path} is not a directory."
    _register_catalog(LocalCatalog(location=catalog_path))


def _catalogs_list():
    return list(Artifactories())


def _register_all_catalogs():
    _register_catalog(GithubCatalog())
    _register_catalog(LocalCatalog())
    _reset_env_local_catalogs()


def _reset_env_local_catalogs():
    for catalog in _catalogs_list():
        if isinstance(catalog, EnvironmentLocalCatalog):
            _unregister_catalog(catalog)
    if UNITXT_ARTIFACTORIES_ENV_VAR in os.environ:
        for path in os.environ[UNITXT_ARTIFACTORIES_ENV_VAR].split(PATHS_SEP):
            _register_catalog(EnvironmentLocalCatalog(location=path))


def _register_all_artifacts():
    dir = os.path.dirname(__file__)
    file_name = os.path.basename(__file__)

    for file in os.listdir(dir):
        if (
            file.endswith(".py")
            and file not in non_registered_files
            and file != file_name
        ):
            module_name = file.replace(".py", "")

            module = importlib.import_module("." + module_name, __package__)

            for _name, obj in inspect.getmembers(module):
                # Make sure the object is a class
                if inspect.isclass(obj):
                    # Make sure the class is a subclass of Artifact (but not Artifact itself)
                    if issubclass(obj, Artifact) and obj is not Artifact:
                        Artifact.register_class(obj)


class ProjectArtifactRegisterer(metaclass=Singleton):
    def __init__(self):
        if not hasattr(self, "_registered"):
            self._registered = False

        if not self._registered:
            _register_all_catalogs()
            _register_all_artifacts()
            self._registered = True


def register_all_artifacts():
    ProjectArtifactRegisterer()
