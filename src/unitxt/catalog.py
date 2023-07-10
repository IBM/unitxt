from .artifact import Artifact, Artifactory, register_atrifactory
from .file_utils import get_all_files_in_dir

import os

class Catalog(Artifactory):
    name: str = None
    location: str = None

catalog_path = os.path.dirname(__file__) + "/catalog"

class LocalCatalog(Catalog):
    name: str = "local"
    location: str = catalog_path

    @property
    def path_dict(self):
        result = {}
        for path in get_all_files_in_dir(self.location, recursive=True, file_extension=".json"):
            name = os.path.splitext(os.path.basename(path))[0]
            result[name] = path
        return result

    def path(self, artifact_identifier: str):
        return self.path_dict.get(artifact_identifier, None)

    def load(self, artifact_identifier: str):
        assert artifact_identifier in self, "Artifact with name {} does not exist".format(artifact_identifier)
        path = self.path(artifact_identifier)
        artifact_instance = Artifact.load(path)
        return artifact_instance

    def __getitem__(self, name) -> Artifact:
        return self.load(name)

    def __contains__(self, artifact_identifier: str):
        if not os.path.exists(self.location):
            return False
        path = self.path(artifact_identifier)
        if path is None:
            return False
        return os.path.exists(path) and os.path.isfile(path)

    def save(self, artifact: Artifact, artifact_identifier: str, collection: str, overwrite: bool = False):
        assert isinstance(artifact, Artifact), "Artifact must be an instance of Artifact"

        if not overwrite:
            assert (
                artifact_identifier not in self
            ), f"Artifact with name {artifact_identifier} already exists in catalog {self.name}"

        collection_dir = os.path.join(self.location, collection)
        os.makedirs(collection_dir, exist_ok=True)
        path = os.path.join(collection_dir, artifact_identifier + ".json")
        artifact.save(path)


register_atrifactory(LocalCatalog())

try:
    import unitxt

    library_catalog = LocalCatalog("library", unitxt.__path__[0] + "/catalog")
    register_atrifactory(library_catalog)
except:
    pass
# create a catalog for the community


class CommunityCatalog(Catalog):
    name = "community"
    location = "https://raw.githubusercontent.com/unitxt/unitxt/main/catalog/community.json"

    def load(self, artifact_identifier: str):
        pass


def add_to_catalog(artifact: Artifact, name: str, collection=str, catalog: Catalog = None, overwrite: bool = False):
    if catalog is None:
        catalog = LocalCatalog()
    catalog.save(artifact, name, collection, overwrite=overwrite)
