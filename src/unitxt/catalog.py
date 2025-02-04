import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import requests

from .artifact import (
    AbstractCatalog,
    Artifact,
    ArtifactLink,
    Catalogs,
    get_catalog_name_and_args,
    reset_artifacts_json_cache,
    verify_legal_catalog_name,
)
from .logging_utils import get_logger
from .settings_utils import get_constants
from .text_utils import print_dict
from .version import version

logger = get_logger()
constants = get_constants()


class Catalog(AbstractCatalog):
    name: str = None
    location: str = None

    def __repr__(self):
        return f"{self.location}"


class LocalCatalog(Catalog):
    name: str = "local"
    location: str = constants.default_catalog_path
    is_local: bool = True

    def path(self, artifact_identifier: str):
        assert (
            artifact_identifier.strip()
        ), "artifact_identifier should not be an empty string."
        parts = artifact_identifier.split(constants.catalog_hierarchy_sep)
        parts[-1] = parts[-1] + ".json"
        return os.path.join(self.location, *parts)

    def load(self, artifact_identifier: str, overwrite_args=None):
        assert (
            artifact_identifier in self
        ), f"Artifact with name {artifact_identifier} does not exist"
        path = self.path(artifact_identifier)
        return Artifact.load(
            path,
            artifact_identifier=artifact_identifier,
            overwrite_args=overwrite_args,
        )

    def __getitem__(self, name) -> Artifact:
        return self.load(name)

    def get_with_overwrite(self, name, overwrite_args):
        return self.load(name, overwrite_args=overwrite_args)

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

    def load(self, artifact_identifier: str, overwrite_args=None):
        url = self.path(artifact_identifier)
        response = requests.get(url)
        data = response.json()
        new_artifact = Artifact.from_dict(data, overwrite_args=overwrite_args)
        new_artifact.__id__ = artifact_identifier
        return new_artifact

    def __contains__(self, artifact_identifier: str):
        url = self.path(artifact_identifier)
        response = requests.head(url)
        return response.status_code == 200


def add_to_catalog(
    artifact: Artifact,
    name: str,
    catalog: Catalog = None,
    overwrite: bool = False,
    catalog_path: Optional[str] = None,
    verbose=True,
):
    reset_artifacts_json_cache()
    if catalog is None:
        if catalog_path is None:
            catalog_path = constants.default_catalog_path
        catalog = LocalCatalog(location=catalog_path)
    verify_legal_catalog_name(name)
    catalog.save_artifact(artifact, name, overwrite=overwrite, verbose=verbose)


def add_link_to_catalog(
    artifact_linked_to: str,
    name: str,
    deprecate: bool = False,
    catalog: Catalog = None,
    overwrite: bool = False,
    catalog_path: Optional[str] = None,
    verbose=True,
):
    if deprecate:
        deprecated_msg = f"Artifact '{name}' is deprecated. Artifact '{artifact_linked_to}' will be instantiated instead. "
        deprecated_msg += f"In future uses, please reference artifact '{artifact_linked_to}' directly."
    else:
        deprecated_msg = None

    artifact_link = ArtifactLink(
        to=artifact_linked_to, __deprecated_msg__=deprecated_msg
    )

    add_to_catalog(
        artifact=artifact_link,
        name=name,
        catalog=catalog,
        overwrite=overwrite,
        catalog_path=catalog_path,
        verbose=verbose,
    )


def get_from_catalog(
    name: str,
    catalog: Catalog = None,
    catalog_path: Optional[str] = None,
):
    if catalog_path is not None:
        catalog = LocalCatalog(location=catalog_path)

    if catalog is None:
        catalogs = None
    else:
        catalogs = [catalog]

    catalog, name, args = get_catalog_name_and_args(name, catalogs=catalogs)

    return catalog.get_with_overwrite(
        name=name,
        overwrite_args=args,
    )


def get_local_catalogs_paths():
    result = []
    for catalog in Catalogs():
        if isinstance(catalog, LocalCatalog):
            if catalog.is_local:
                result.append(catalog.location)
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
    done = set()
    for local_catalog_path in get_local_catalogs_paths():
        if local_catalog_path not in done:
            result += Counter(local_catalog_summary(local_catalog_path))
        done.add(local_catalog_path)
    print_dict(result)
    return result


def _get_tags_from_file(file_path):
    result = Counter()

    with open(file_path) as f:
        data = json.load(f)
        if "__tags__" in data and isinstance(data["__tags__"], dict):
            tags = data["__tags__"]
            for key, value in tags.items():
                if isinstance(value, list):
                    for item in value:
                        result[f"{key}:{item}"] += 1
                else:
                    result[f"{key}:{value}"] += 1

    return result


def count_tags():
    result = Counter()
    done = set()

    for local_catalog_path in get_local_catalogs_paths():
        if local_catalog_path not in done:
            for root, _, files in os.walk(local_catalog_path):
                for file in files:
                    if file.endswith(".json"):
                        file_path = os.path.join(root, file)
                        try:
                            result += _get_tags_from_file(file_path)
                        except json.JSONDecodeError:
                            logger.info(f"Error decoding JSON in file: {file_path}")
                        except OSError:
                            logger.info(f"Error reading file: {file_path}")

            done.add(local_catalog_path)

    print_dict(result)
    return result


def ls(to_file=None):
    done = set()
    result = []
    for local_catalog_path in get_local_catalogs_paths():
        if local_catalog_path not in done:
            for root, _, files in os.walk(local_catalog_path):
                for file in files:
                    if ".json" not in file:
                        continue
                    file_path = os.path.relpath(
                        os.path.join(root, file), local_catalog_path
                    )
                    file_id = ".".join(
                        file_path.replace(".json", "").split(os.path.sep)
                    )
                    result.append(file_id)
    if to_file:
        with open(to_file, "w+") as f:
            f.write("\n".join(result))
    else:
        logger.info("\n".join(result))
    return result
