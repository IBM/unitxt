from unitxt.artifact import Artifact

from tests.utils import UnitxtTestCase


class TestArtifactRegistration(UnitxtTestCase):
    def test_subclass_registration(self):
        class DummyShouldBeRegistered(Artifact):
            pass

        assert Artifact.is_registered_type("dummy_should_be_registered")
        assert Artifact.is_registered_class(DummyShouldBeRegistered)
