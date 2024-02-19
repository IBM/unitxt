from src.unitxt.artifact import (
    Artifact,
    fetch_artifact,
)
from src.unitxt.catalog import add_to_catalog, get_from_catalog
from src.unitxt.dataclass import UnexpectedArgumentError
from src.unitxt.logging_utils import get_logger
from src.unitxt.operator import SequentialOperator
from src.unitxt.processors import StringOrNotString
from src.unitxt.test_utils.catalog import temp_catalog
from tests.utils import UnitxtTestCase

logger = get_logger()


class TestArtifact(UnitxtTestCase):
    def test_artifact_identifier_setter(self):
        artifact = Artifact()
        artifact_identifier = "artifact.id.dummy"
        artifact.artifact_identifier = artifact_identifier
        self.assertEqual(artifact_identifier, artifact.artifact_identifier)

    def test_artifact_identifier_cannot_be_used_as_keyword_arg(self):
        """Test that artifact_identifier cannot be set in construction.

        Since it is an internal field, and isn't serialized, it should never be set when
        constructing an Artifact from kwargs.
        """
        with self.assertRaises(UnexpectedArgumentError):
            Artifact(artifact_identifier="artifact.id.dummy")

    def test_artifact_identifier_available_for_loaded_artifacts(self):
        artifact_identifier = "tasks.classification.binary"
        artifact, _ = fetch_artifact(artifact_identifier)
        self.assertEqual(artifact_identifier, artifact.artifact_identifier)

    def test_artifact_loading_with_overwrite_args(self):
        with temp_catalog() as catalog_path:
            add_to_catalog(
                StringOrNotString(string="yes", field="a_field"),
                "test1.test2",
                catalog_path=catalog_path,
            )
            artifact = get_from_catalog(
                "test1.test2[string=hello]", catalog_path=catalog_path
            )
            self.assertEqual(artifact.string, "hello")

    def test_artifact_loading_with_overwrite_args_with_list_of_operators(self):
        with temp_catalog() as catalog_path:
            add_to_catalog(
                StringOrNotString(string="yes", field="a_field"),
                "test2.processor",
                catalog_path=catalog_path,
            )
            add_to_catalog(
                SequentialOperator(),
                "test2.seq",
                catalog_path=catalog_path,
            )
            artifact = get_from_catalog(
                "test2.seq[steps=[test2.processor[string=no]]]",
                catalog_path=catalog_path,
            )
            self.assertEqual(artifact.steps[0].string, "no")

    def test_artifact_loading_with_overwrite_args_list(self):
        artifact_identifier = (
            "tasks.classification.binary[metrics=[metrics.rouge, metrics.accuracy]]"
        )
        artifact, _ = fetch_artifact(artifact_identifier)
        self.assertEqual(artifact.metrics, ["metrics.rouge", "metrics.accuracy"])
