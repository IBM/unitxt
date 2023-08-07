import json
import tempfile

from .. import add_to_catalog, register_local_catalog
from ..artifact import fetch_artifact
from ..text_utils import print_dict

TEMP_NAME = "tmp_name"


def test_artfifact_saving_and_loading(artifact, tester=None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        add_to_catalog(artifact, TEMP_NAME, overwrite=True, catalog_path=tmp_dir)
        register_local_catalog(tmp_dir)
        loaded_artifact, _ = fetch_artifact(TEMP_NAME)
        if tester is not None:
            with tester.subTest(artifact=artifact, loaded_artifact=loaded_artifact):
                tester.assertDictEqual(loaded_artifact.to_dict(), artifact.to_dict())
        else:
            if not json.dumps(loaded_artifact.to_dict(), sort_keys=True) == json.dumps(
                artifact.to_dict(), sort_keys=True
            ):
                print("Artifact loaded is not equal to artifact stored")
                print_dict(loaded_artifact.to_dict())
                print_dict(artifact.to_dict())
                raise AssertionError("Artifact loaded is not equal to artifact stored")
