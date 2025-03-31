from unitxt.operator import MissingRequirementsError, PackageRequirementsMixin

from tests.utils import UnitxtTestCase


class TestPackageRequirementsMixin(UnitxtTestCase):
    def setUp(self):
        # Subclass of PackageRequirementsMixin to test without needing an actual Artifact class
        class TestArtifact(PackageRequirementsMixin):
            pass

        self.artifact = TestArtifact()

    def test_missing_package(self):
        """Test case where a package is missing."""
        self.artifact._requirements_list = ["nonexistent_package"]
        with self.assertRaises(MissingRequirementsError) as cm:
            self.artifact.check_missing_requirements()
        error = cm.exception
        self.assertIn("nonexistent_package", error.missing_packages)
        self.assertIn("nonexistent_package", error.message)

    def test_version_mismatch(self):
        """Test case where a package has a version conflict."""
        self.artifact._requirements_list = ["datasets>=500.2.4"]
        with self.assertRaises(MissingRequirementsError) as cm:
            self.artifact.check_missing_requirements()
        error = cm.exception
        self.assertIn("datasets>=500.2.4", error.version_mismatched_packages[0])
        self.assertIn("datasets", error.message)

    def test_requirements_with_custom_instructions(self):
        """Test case with custom installation instructions for missing packages."""
        self.artifact._requirements_list = {
            "nonexistent_package": "Install with `pip install nonexistent_package`"
        }
        with self.assertRaises(MissingRequirementsError) as cm:
            self.artifact.check_missing_requirements()
        error = cm.exception
        self.assertIn("nonexistent_package", error.missing_packages)
        self.assertIn(
            "Install with `pip install nonexistent_package`",
            error.installation_instructions[0],
        )
