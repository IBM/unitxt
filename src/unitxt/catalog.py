import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import requests

from .artifact import Artifact, Artifactories, Artifactory, reset_artifacts_cache
from .logging_utils import get_logger
from .text_utils import print_dict
from .version import version

logger = get_logger()
COLLECTION_SEPARATOR = "."
PATHS_SEP = ":"


class Catalog(Artifactory):
    name: str = None
    location: str = None


try:
    import unitxt

    if unitxt.__file__:
        lib_dir = os.path.dirname(unitxt.__file__)
    else:
        lib_dir = os.path.dirname(__file__)
except ImportError:
    lib_dir = os.path.dirname(__file__)

default_catalog_path = os.path.join(lib_dir, "catalog")


class LocalCatalog(Catalog):
    name: str = "local"
    location: str = default_catalog_path
    is_local: bool = True

    def path(self, artifact_identifier: str):
        assert (
            artifact_identifier.strip()
        ), "artifact_identifier should not be an empty string."
        parts = artifact_identifier.split(COLLECTION_SEPARATOR)
        parts[-1] = parts[-1] + ".json"
        return os.path.join(self.location, *parts)

    def load(self, artifact_identifier: str):
        assert (
            artifact_identifier in self
        ), f"Artifact with name {artifact_identifier} does not exist"
        path = self.path(artifact_identifier)
        return Artifact.load(path, artifact_identifier)

    def __getitem__(self, name) -> Artifact:
        return self.load(name)

    def __contains__(self, artifact_identifier: str):
        if not os.path.exists(self.location):
            return False
        path = self.path(artifact_identifier)
        if path is None:
            return False
        return os.path.exists(path) and os.path.isfile(path)

    def save_artifact(
        self,
        artifact: Artifact,
        artifact_identifier: str,
        overwrite: bool = False,
        verbose: bool = True,
    ):
        assert isinstance(
            artifact, Artifact
        ), f"Input artifact must be an instance of Artifact, got {type(artifact)}"
        if not overwrite:
            assert (
                artifact_identifier not in self
            ), f"Artifact with name {artifact_identifier} already exists in catalog {self.name}"
        path = self.path(artifact_identifier)
        os.makedirs(Path(path).parent.absolute(), exist_ok=True)
        artifact.save(path)
        if verbose:
            logger.info(f"Artifact {artifact_identifier} saved to {path}")


class EnvironmentLocalCatalog(LocalCatalog):
    pass


class GithubCatalog(LocalCatalog):
    name = "community"
    repo = "unitxt"
    repo_dir = "src/unitxt/catalog"
    user = "IBM"
    is_local: bool = False

    def prepare(self):
        tag = version
        self.location = f"https://raw.githubusercontent.com/{self.user}/{self.repo}/{tag}/{self.repo_dir}"

    def load(self, artifact_identifier: str):
        url = self.path(artifact_identifier)
        response = requests.get(url)
        data = response.json()
        new_artifact = Artifact.from_dict(data)
        new_artifact.artifact_identifier = artifact_identifier
        return new_artifact

    def __contains__(self, artifact_identifier: str):
        url = self.path(artifact_identifier)
        response = requests.head(url)
        return response.status_code == 200


def verify_legal_catalog_name(name):
    assert re.match(
        r"^[\w" + COLLECTION_SEPARATOR + "]+$", name
    ), f'Artifict name ("{name}") should be alphanumeric. Use "." for nesting (e.g. myfolder.my_artifact)'


def add_to_catalog(
    artifact: Artifact,
    name: str,
    catalog: Catalog = None,
    overwrite: bool = False,
    catalog_path: Optional[str] = None,
    verbose=True,
):
    reset_artifacts_cache()
    if catalog is None:
        if catalog_path is None:
            catalog_path = default_catalog_path
        catalog = LocalCatalog(location=catalog_path)
    verify_legal_catalog_name(name)
    catalog.save_artifact(
        artifact, name, overwrite=overwrite, verbose=verbose
    )  # remove collection (its actually the dir).
    # verify name


def get_local_catalogs_paths():
    result = []
    for artifactory in Artifactories():
        if isinstance(artifactory, LocalCatalog):
            if artifactory.is_local:
                result.append(artifactory.location)
    return result


def count_files_recursively(folder):
    file_count = 0
    for _, _, files in os.walk(folder):
        file_count += len(files)
    return file_count


def local_catalog_summary(catalog_path):
    result = {}

    for dir in os.listdir(catalog_path):
        if os.path.isdir(os.path.join(catalog_path, dir)):
            result[dir] = count_files_recursively(os.path.join(catalog_path, dir))

    return result


def summary():
    result = Counter()
    for local_catalog_path in get_local_catalogs_paths():
        result += Counter(local_catalog_summary(local_catalog_path))
    print_dict(result)
    return result
