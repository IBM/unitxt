import unittest

from src.unitxt.artifact import Artifact


class TestArtifactRegistration(unittest.TestCase):
    def test_subclass_registration(self):
        class DummyShouldBeRegistered(Artifact):
            pass

        assert Artifact.is_registered_type("dummy_should_be_registered")
        assert Artifact.is_registered_class(DummyShouldBeRegistered)
