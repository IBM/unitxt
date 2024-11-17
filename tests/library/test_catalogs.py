import json
import os
import tempfile

import unitxt
from unitxt import add_to_catalog
from unitxt.artifact import Artifact, Catalogs
from unitxt.error_utils import UnitxtError
from unitxt.register import (
    _reset_env_local_catalogs,
    register_local_catalog,
    unregister_local_catalog,
)

from tests.utils import UnitxtTestCase


class TestCatalogs(UnitxtTestCase):
    def test_catalog_registration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            register_local_catalog(tmp_dir)
            artifs = Catalogs()
            first_artif = next(iter(artifs))
            self.assertTrue(isinstance(first_artif, unitxt.catalog.LocalCatalog))
            self.assertEqual(first_artif.location, tmp_dir)

    def test_catalog_unregistration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            register_local_catalog(tmp_dir)
            unregister_local_catalog(tmp_dir)
            artifs = Catalogs()
            first_artif = next(iter(artifs))
            self.assertNotEqual(first_artif.location, tmp_dir)

    def test_catalog_registration_with_env_var(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["UNITXT_CATALOGS"] = tmp_dir
            _reset_env_local_catalogs()
            artifs = Catalogs()
            first_artif = next(iter(artifs))
            self.assertTrue(isinstance(first_artif, unitxt.catalog.LocalCatalog))
            self.assertEqual(first_artif.location, tmp_dir)
            del os.environ["UNITXT_CATALOGS"]

    def test_catalog_registration_with_both_env_var(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["UNITXT_ARTIFACTORIES"] = tmp_dir
            os.environ["UNITXT_CATALOGS"] = tmp_dir

            with self.assertRaises(UnitxtError) as cm:
                _reset_env_local_catalogs()

            self.assertTrue(
                "Both UNITXT_CATALOGS and UNITXT_ARTIFACTORIES are set."
                in str(cm.exception)
            )

            del os.environ["UNITXT_ARTIFACTORIES"]
            del os.environ["UNITXT_CATALOGS"]

    def test_catalog_registration_with_old_env_var(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["UNITXT_ARTIFACTORIES"] = tmp_dir
            _reset_env_local_catalogs()
            artifs = Catalogs()
            first_artif = next(iter(artifs))
            self.assertTrue(isinstance(first_artif, unitxt.catalog.LocalCatalog))
            self.assertEqual(first_artif.location, tmp_dir)
            del os.environ["UNITXT_ARTIFACTORIES"]

    def test_add_to_catalog(self):
        with tempfile.TemporaryDirectory() as tmp_dir:

            class ClassToSave(Artifact):
                t: int = 0

            add_to_catalog(ClassToSave(t=1), "test.save", catalog_path=tmp_dir)

            with open(os.path.join(tmp_dir, "test", "save.json")) as f:
                content = json.load(f)

            self.assertDictEqual(content, {"__type__": "class_to_save", "t": 1})
