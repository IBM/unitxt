import os
from pathlib import Path


def add_local_catalog_to_artifactories_env_var():
    local_catalog_path = os.path.join(Path(__file__).parent.parent, "catalog")
    paths = [local_catalog_path]
    if "UNITXT_ARTIFACTORIES" in os.environ:
        paths = os.environ["UNITXT_ARTIFACTORIES"].split(":")
        if local_catalog_path not in paths:
            paths.insert(0, local_catalog_path)
            os.environ["UNITXT_ARTIFACTORIES"] = ":".join(paths)
        # else do nothing, since local_catalog_path is already in paths
    else:
        os.environ["UNITXT_ARTIFACTORIES"] = local_catalog_path
