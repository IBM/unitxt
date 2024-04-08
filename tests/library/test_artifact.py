from unitxt.artifact import (
    Artifact,
    fetch_artifact,
    reset_artifacts_json_cache,
)
from unitxt.catalog import add_to_catalog, get_from_catalog
from unitxt.dataclass import UnexpectedArgumentError
from unitxt.logging_utils import get_logger
from unitxt.operator import SequentialOperator
from unitxt.operators import AddFields, RenameFields
from unitxt.processors import StringOrNotString
from unitxt.test_utils.catalog import temp_catalog

from tests.utils import UnitxtTestCase

logger = get_logger()


class TestArtifact(UnitxtTestCase):
    def test_artifact_identifier_setter(self):
        artifact = Artifact()
        artifact_identifier = "artifact.id.dummy"
        artifact.__id__ = artifact_identifier
        self.assertEqual(artifact_identifier, artifact.__id__)

    def test_artifact_identifier_cannot_be_used_as_keyword_arg(self):
        """Test that artifact_identifier cannot be set in construction.

        Since it is an internal field, and isn't serialized, it should never be set when
        constructing an Artifact from kwargs.
        """
        with self.assertRaises(UnexpectedArgumentError):
            Artifact(__id__="artifact.id.dummy")

    def test_artifact_identifier_available_for_loaded_artifacts(self):
        artifact_identifier = "tasks.classification.binary"
        artifact, _ = fetch_artifact(artifact_identifier)
        self.assertEqual(artifact_identifier, artifact.__id__)

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

    def test_artifact_loading_with_overwrite_args_dict(self):
        with temp_catalog() as catalog_path:
            add_to_catalog(
                AddFields(
                    fields={
                        "classes": ["war", "peace"],
                        "text_type": "text",
                        "type_of_class": "topic",
                    }
                ),
                "addfields.for.test.dict",
                catalog_path=catalog_path,
            )
            add_to_catalog(
                RenameFields(field_to_field={"label_text": "label"}),
                "renamefields.for.test.dict",
                catalog_path=catalog_path,
            )
            artifact = get_from_catalog(
                "addfields.for.test.dict",
                catalog_path=catalog_path,
            )
            expected = {
                "classes": ["war", "peace"],
                "text_type": "text",
                "type_of_class": "topic",
            }
            self.assertDictEqual(expected, artifact.fields)

            # with overwrite
            artifact = get_from_catalog(
                "addfields.for.test.dict[fields={classes=[war_test, peace_test],text_type= text_test, type_of_class= topic_test}]",
                catalog_path=catalog_path,
            )
            expected = {
                "classes": ["war_test", "peace_test"],
                "text_type": "text_test",
                "type_of_class": "topic_test",
            }
            self.assertDictEqual(expected, artifact.fields)

    def test_modifying_fetched_artifact_does_not_effect_cached_artifacts(self):
        artifact_identifier = "metrics.accuracy"
        artifact, artifactory1 = fetch_artifact(artifact_identifier)
        self.assertNotEqual(artifact.n_resamples, None)
        artifact.disable_confidence_interval_calculation()
        self.assertEqual(artifact.n_resamples, None)

        same_artifact_retrieved_again, artifactory2 = fetch_artifact(
            artifact_identifier
        )
        self.assertNotEqual(same_artifact_retrieved_again.n_resamples, None)

        # returned artifactories should be the same object
        self.assertTrue(artifactory1 == artifactory2)

    def test_reset_artifacts_json_cache(self):
        reset_artifacts_json_cache()
