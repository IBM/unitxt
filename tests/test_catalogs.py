import unittest
from src import unitxt
from src.unitxt import register_local_catalog
from src.unitxt.artifact import Artifactories
import tempfile

class TestCatalogs(unittest.TestCase):
    
    def test_catalog_registration(self):
        tmp_dir = tempfile.gettempdir()
        register_local_catalog(tmp_dir)
        artifs = Artifactories()
        first_artif = next(iter(artifs))
        self.assertTrue(isinstance(first_artif, unitxt.catalog.LocalCatalog))
        self.assertEqual(first_artif.location, tmp_dir)
