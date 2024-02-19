import os
import tempfile

from src import unitxt
from src.unitxt import register_local_catalog
from src.unitxt.artifact import Artifactories
from src.unitxt.register import _reset_env_local_catalogs
from tests.utils import UnitxtTestCase


class TestCatalogs(UnitxtTestCase):
    def test_catalog_registration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            register_local_catalog(tmp_dir)
            artifs = Artifactories()
            first_artif = next(iter(artifs))
            self.assertTrue(isinstance(first_artif, unitxt.catalog.LocalCatalog))
            self.assertEqual(first_artif.location, tmp_dir)

    def test_catalog_registration_with_env_var(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["UNITXT_ARTIFACTORIES"] = tmp_dir
            _reset_env_local_catalogs()
            artifs = Artifactories()
            first_artif = next(iter(artifs))
            self.assertTrue(isinstance(first_artif, unitxt.catalog.LocalCatalog))
            self.assertEqual(first_artif.location, tmp_dir)
