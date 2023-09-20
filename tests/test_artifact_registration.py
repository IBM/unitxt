import unittest

from src.unitxt.artifact import Artifact


class TestArtifactRegistration(unittest.TestCase):
    def test_subclass_registration(self):
        class DummyExist(Artifact):
            pass

        assert Artifact.is_registered_type("dummy_exist")
        assert Artifact.is_registered_class(DummyExist)
