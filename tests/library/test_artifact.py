import json
import os

from unitxt.artifact import (
    Artifact,
    fetch_artifact,
    get_artifacts_data_classification,
    reset_artifacts_json_cache,
)
from unitxt.catalog import add_to_catalog, get_from_catalog
from unitxt.dataclass import UnexpectedArgumentError
from unitxt.error_utils import UnitxtError
from unitxt.logging_utils import get_logger
from unitxt.metrics import Accuracy, F1Binary
from unitxt.operator import SequentialOperator
from unitxt.operators import Rename, Set
from unitxt.processors import StringEquals
from unitxt.settings_utils import get_settings
from unitxt.templates import YesNoTemplate
from unitxt.test_utils.catalog import temp_catalog

from tests.utils import UnitxtTestCase

logger = get_logger()
settings = get_settings()


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
                StringEquals(string="yes", field="a_field"),
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
                StringEquals(string="yes", field="a_field"),
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
        artifact_identifier = "tasks.classification.binary.zero_or_one[metrics=[metrics.roc_auc, metrics.accuracy]]"
        artifact, _ = fetch_artifact(artifact_identifier)
        self.assertEqual(artifact.metrics, ["metrics.roc_auc", "metrics.accuracy"])

    def test_artifact_loading_with_overwrite_args_dict(self):
        with temp_catalog() as catalog_path:
            add_to_catalog(
                Set(
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
                Rename(field_to_field={"label_text": "label"}),
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

    def test_checking_data_classification_policy_env(self):
        artifact_name = "metrics.accuracy"
        expected_data_classification_policies = {artifact_name: ["public"]}
        os.environ["UNITXT_DATA_CLASSIFICATION_POLICY"] = json.dumps(
            expected_data_classification_policies
        )

        data_classification_policies = get_artifacts_data_classification(artifact_name)
        self.assertListEqual(
            expected_data_classification_policies[artifact_name],
            data_classification_policies,
        )

        metric = Accuracy()
        metric.__id__ = artifact_name
        instance = {"data": "text", "data_classification_policy": ["public"]}
        output = metric.verify_instance(instance)
        self.assertEqual(instance, output)

        instance["data_classification_policy"] = ["pii"]
        with self.assertRaises(UnitxtError) as e:
            metric.verify_instance(instance)
        self.assertEqual(
            str(e.exception),
            f"The instance '{instance} 'has the following data classification policy "
            f"'{instance['data_classification_policy']}', however, the artifact "
            f"'{artifact_name}' is only configured to support the data with classification "
            f"'{data_classification_policies}'. To enable this either change "
            f"the 'data_classification_policy' attribute of the artifact, "
            f"or modify the environment variable "
            f"'UNITXT_DATA_CLASSIFICATION_POLICY' accordingly.\nFor more information: see https://www.unitxt.ai/en/latest//docs/data_classification_policy.html \n",
        )
        # "Fixing" the env variable so that it does not affect other tests:
        del os.environ["UNITXT_DATA_CLASSIFICATION_POLICY"]

    def test_checking_data_classification_policy_attribute(self):
        instance = {"data": "text", "data_classification_policy": ["public"]}
        metric = F1Binary(data_classification_policy=["public"])
        output = metric.verify_instance(instance)
        self.assertEqual(instance, output)

        template = YesNoTemplate(data_classification_policy=["propriety", "pii"])
        instance["data_classification_policy"] = ["propriety"]
        output = template.verify_instance(instance)
        self.assertEqual(instance, output)

    def test_misconfigured_data_classification_policy(self):
        wrong_data_classification = "public"

        os.environ["UNITXT_DATA_CLASSIFICATION_POLICY"] = wrong_data_classification
        with self.assertRaises(RuntimeError) as e:
            get_artifacts_data_classification("")
        self.assertEqual(
            str(e.exception),
            f"If specified, the value of 'UNITXT_DATA_CLASSIFICATION_POLICY' "
            f"should be a valid json dictionary. Got '{wrong_data_classification}' "
            f"instead.",
        )

        with self.assertRaises(ValueError) as e:
            Accuracy(data_classification_policy=wrong_data_classification)
        self.assertEqual(
            str(e.exception),
            f"The 'data_classification_policy' of Accuracy must be either None - "
            f"in case when no policy applies - or a list of strings, for example: ['public']. "
            f"However, '{wrong_data_classification}' of type {type(wrong_data_classification)} was provided instead.",
        )

        # "Fixing" the env variable so that it does not affect other tests:
        del os.environ["UNITXT_DATA_CLASSIFICATION_POLICY"]
