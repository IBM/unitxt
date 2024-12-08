import json
import os

from unitxt.artifact import (
    Artifact,
    ArtifactLink,
    fetch_artifact,
    get_artifacts_data_classification,
    reset_artifacts_json_cache,
)
from unitxt.card import TaskCard
from unitxt.catalog import add_link_to_catalog, add_to_catalog, get_from_catalog
from unitxt.dataclass import UnexpectedArgumentError
from unitxt.error_utils import UnitxtError
from unitxt.loaders import LoadFromDictionary
from unitxt.logging_utils import get_logger
from unitxt.metrics import Accuracy, F1Binary
from unitxt.operator import SequentialOperator
from unitxt.operators import Copy, Rename, Set
from unitxt.processors import StringEquals
from unitxt.settings_utils import get_settings
from unitxt.task import Task
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

    def test_artifact_not_found(self):
        artifact_identifier = "artifact.not_found"
        with self.assertRaises(UnitxtError) as cm:
            artifact, _ = fetch_artifact(artifact_identifier)
        self.assertTrue(
            "Artifact artifact.not_found does not exist, in Unitxt catalogs"
            in str(cm.exception)
        )

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
        artifact, catalog1 = fetch_artifact(artifact_identifier)
        self.assertNotEqual(artifact.n_resamples, None)
        artifact.disable_confidence_interval_calculation()
        self.assertEqual(artifact.n_resamples, None)

        same_artifact_retrieved_again, catalog2 = fetch_artifact(artifact_identifier)
        self.assertNotEqual(same_artifact_retrieved_again.n_resamples, None)

        # returned catalogs should be the same object
        self.assertTrue(catalog1 == catalog2)

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

    def test_artifact_link(self):
        rename = Rename(field_to_field={"old_field_name": "new_field_name"})
        with temp_catalog() as catalog_path:
            add_to_catalog(
                rename,
                "rename.for.test.artifact.link",
                catalog_path=catalog_path,
                overwrite=True,
            )

            rename_fields = ArtifactLink("rename.for.test.artifact.link")
            add_to_catalog(
                rename_fields,
                "renamefields.for.test.artifact.link",
                catalog_path=catalog_path,
                overwrite=True,
            )

            artifact = get_from_catalog("renamefields.for.test.artifact.link")
            self.assertDictEqual(rename.to_dict(), artifact.to_dict())

            # test again, now employing add_link_to_catalog()
            add_link_to_catalog(
                artifact_linked_to="rename.for.test.artifact.link",
                name="renamefields2",
                catalog_path=catalog_path,
                overwrite=True,
            )
            artifact = get_from_catalog("renamefields2")
            self.assertDictEqual(rename.to_dict(), artifact.to_dict())

    def test_artifact_link_from_artifactlink_object(self):
        rename = Rename(field_to_field={"old_field_name": "new_field_name"})
        with temp_catalog() as catalog_path:
            add_to_catalog(
                rename,
                "rename.for.test.artifact.link",
                catalog_path=catalog_path,
                overwrite=True,
            )

            rename_fields = ArtifactLink("rename.for.test.artifact.link")

            artifact, _ = fetch_artifact(rename_fields)
            self.assertDictEqual(rename.to_dict(), artifact.to_dict())

    def test_artifact_link_to_link(self):
        rename = Rename(field_to_field={"old_field_name": "new_field_name"})
        with temp_catalog() as catalog_path:
            add_to_catalog(
                rename,
                "rename.for.test.artifact.link",
                catalog_path=catalog_path,
                overwrite=True,
            )

            rename_fields = ArtifactLink("rename.for.test.artifact.link")
            add_to_catalog(
                rename_fields,
                "renamefields.for.test.artifact.link",
                catalog_path=catalog_path,
                overwrite=True,
            )

            rename_fields_junior = ArtifactLink(
                "renamefields.for.test.artifact.link",
            )
            add_to_catalog(
                rename_fields_junior,
                "renamefields.junior.for.test.artifact.link",
                catalog_path=catalog_path,
                overwrite=True,
            )

            rename_fields_junior_junior = ArtifactLink(
                "renamefields.junior.for.test.artifact.link",
            )
            add_to_catalog(
                rename_fields_junior_junior,
                "renamefields.junior.junior.for.test.artifact.link",
                catalog_path=catalog_path,
                overwrite=True,
            )

            artifact = get_from_catalog(
                "renamefields.junior.junior.for.test.artifact.link"
            )
            self.assertDictEqual(rename.to_dict(), artifact.to_dict())

    def test_artifact_link_with_deprecation_warning(self):
        with temp_catalog() as catalog_path:
            rename = Rename(field_to_field={"old_field_name": "new_field_name"})
            add_to_catalog(
                rename,
                "rename.for.test.artifact.link",
                catalog_path=catalog_path,
                overwrite=True,
            )

            with self.assertWarns(DeprecationWarning):
                rename_fields = ArtifactLink(
                    artifact_linked_to="rename.for.test.artifact.link",
                    __deprecated_msg__="Artifact is deprecated. "
                    "'rename.for.test.artifact.link' is now instantiated instead. "
                    "\nIn the future, please use 'rename.for.test.artifact.link'.",
                )

            add_to_catalog(
                rename_fields,
                "renamefields.for.test.artifact.link",
                catalog_path=catalog_path,
                overwrite=True,
            )

            with self.assertWarns(DeprecationWarning):
                artifact, _ = fetch_artifact("renamefields.for.test.artifact.link")
            self.assertDictEqual(rename.to_dict(), artifact.to_dict())

            # test again, now employing add_link_to_catalog()
            add_link_to_catalog(
                artifact_linked_to="rename.for.test.artifact.link",
                name="renamefields3",
                deprecate=True,
                catalog_path=catalog_path,
                overwrite=True,
            )

            with self.assertWarns(DeprecationWarning):
                artifact = get_from_catalog("renamefields3")
            self.assertDictEqual(rename.to_dict(), artifact.to_dict())

    def test_artifact_link_with_overwrites(self):
        with temp_catalog() as catalog_path:
            rename = Rename(field_to_field={"old_name": "new_name"})
            add_to_catalog(
                rename,
                "rename.old_name.to.new_name",
                catalog_path=catalog_path,
                overwrite=True,
            )

            rename_fields = ArtifactLink(
                "rename.old_name.to.new_name",
            )
            add_to_catalog(
                rename_fields,
                "rename_fields.old_name.to.new_name",
                catalog_path=catalog_path,
                overwrite=True,
            )

            overwritten_rename = Rename(
                field_to_field={"overwritten_old_name": "overwritten_new_name"}
            )

            artifact, _ = fetch_artifact(
                "rename_fields.old_name.to.new_name[field_to_field={overwritten_old_name=overwritten_new_name}]"
            )
            self.assertDictEqual(overwritten_rename.to_dict(), artifact.to_dict())

    def test_artifact_link_from_within_overwrites(self):
        with temp_catalog() as catalog_path:
            link_to_metrics_accuracy = ArtifactLink("metrics.accuracy")
            add_to_catalog(
                link_to_metrics_accuracy,
                "link.to.metrics.accuracy",
                catalog_path=catalog_path,
                overwrite=True,
            )

            task_as_is, _ = fetch_artifact("tasks.evaluation.preference")
            task_indirect, _ = fetch_artifact(
                "tasks.evaluation.preference[metrics=[link.to.metrics.accuracy]]"
            )

            # unitxt.task maintains its metrics as a list of str.
            # So currently, the above tasks have different metrics.
            # Now we instantiate them as actual artifacts, and then test that
            # the different names instantiate to same metrics.
            actual_metrics_of_task_as_is = []
            for metric_id in task_as_is.metrics:
                actual_metrics_of_task_as_is.extend(
                    Task.get_metrics_artifacts(metric_id)
                )
            actual_metrics_of_task_indirect = []
            for metric_id in task_indirect.metrics:
                actual_metrics_of_task_indirect.extend(
                    Task.get_metrics_artifacts(metric_id)
                )

            actual_metrics_to_dict_of_task_as_is = [
                metric.to_dict() for metric in actual_metrics_of_task_as_is
            ]
            actual_metrics_to_dict_of_task_indirect = [
                metric.to_dict() for metric in actual_metrics_of_task_indirect
            ]

            self.assertListEqual(
                actual_metrics_to_dict_of_task_as_is,
                actual_metrics_to_dict_of_task_indirect,
            )

    def test_artifact_link_to_link_with_overwrites(self):
        with temp_catalog() as catalog_path:
            rename = Rename(field_to_field={"old_name": "new_name"})
            add_to_catalog(
                rename,
                "rename.old_name.to.new_name",
                catalog_path=catalog_path,
                overwrite=True,
            )

            rename_fields = ArtifactLink(
                "rename.old_name.to.new_name",
            )
            add_to_catalog(
                rename_fields,
                "rename_fields.old_name.to.new_name",
                catalog_path=catalog_path,
                overwrite=True,
            )

            rename_fields_junior = ArtifactLink(
                "rename_fields.old_name.to.new_name",
            )
            add_to_catalog(
                rename_fields_junior,
                "rename_fields.junior.old_name.to.new_name",
                catalog_path=catalog_path,
                overwrite=True,
            )

            rename_fields_junior_junior = ArtifactLink(
                "rename_fields.junior.old_name.to.new_name",
            )
            add_to_catalog(
                rename_fields_junior_junior,
                "rename_fields.junior.junior.old_name.to.new_name",
                catalog_path=catalog_path,
                overwrite=True,
            )

            overwritten_rename = Rename(
                field_to_field={"overwritten_old_name": "overwritten_new_name"}
            )

            artifact, _ = fetch_artifact(
                "rename_fields.junior.junior.old_name.to.new_name[field_to_field={overwritten_old_name=overwritten_new_name}]"
            )
            self.assertDictEqual(overwritten_rename.to_dict(), artifact.to_dict())

    def test_artifact_link_in_recursive_load(self):
        copy_operator = Copy(field_to_field={"a": "b"})
        tiny_card = TaskCard(
            loader=LoadFromDictionary(
                data={"train": [{"a": 1}, {"a": 2}]}
            ),  # any loader would do for this test
            preprocess_steps=[copy_operator],
            task="tasks.generation",  # any task would do for this test
            templates="templates.empty",  # any template would do for this test
        )
        with temp_catalog() as catalog_path:
            add_to_catalog(
                copy_operator,
                "copy.operator",
                catalog_path=catalog_path,
                overwrite=True,
            )

            tiny_card_step_fetched = TaskCard(
                loader=LoadFromDictionary(data={"train": [{"a": 1}, {"a": 2}]}),
                preprocess_steps=["copy.operator"],
                task="tasks.generation",
                templates="templates.empty",
            )
            # verify that tinytiny_card_step_fetched is same as tiny_card, focus on
            # preprocess_steps being the only difference
            self.assertEqual(
                len(tiny_card_step_fetched.preprocess_steps),
                len(tiny_card.preprocess_steps),
            )
            for i in range(len(tiny_card.preprocess_steps)):
                self.assertDictEqual(
                    tiny_card.preprocess_steps[i].to_dict(),
                    tiny_card_step_fetched.preprocess_steps[i].to_dict(),
                )

            link_to_copy_operator = ArtifactLink(
                artifact_linked_to="copy.operator",
            )
            add_to_catalog(
                link_to_copy_operator,
                "link.to.copy.operator",
                catalog_path=catalog_path,
                overwrite=True,
            )
            tiny_card_with_links = TaskCard(
                loader=LoadFromDictionary(data={"train": [{"a": 1}, {"a": 2}]}),
                preprocess_steps=["link.to.copy.operator"],
                task="tasks.generation",
                templates="templates.empty",
            )
            add_to_catalog(
                tiny_card_with_links,
                "tiny.card.with.links",
                catalog_path=catalog_path,
                overwrite=True,
            )

            fetched_tiny_card_with_links, _ = fetch_artifact("tiny.card.with.links")
            # verify that fetched_tiny_card_with_links is same as tiny_card, focus on
            # preprocess_steps being the only difference
            self.assertEqual(
                len(fetched_tiny_card_with_links.preprocess_steps),
                len(tiny_card.preprocess_steps),
            )
            for i in range(len(tiny_card.preprocess_steps)):
                self.assertDictEqual(
                    tiny_card.preprocess_steps[i].to_dict(),
                    fetched_tiny_card_with_links.preprocess_steps[i].to_dict(),
                )

    def test_artifact_is_not_saving_if_artifact_has_changed(self):
        with self.assertRaises(UnitxtError) as e:
            args = {
                "__type__": "standard_recipe",
                "card": "cards.sst2",
                "template_card_index": 0,
                "demos_pool_size": 100,
                "num_demos": 0,
            }
            a = Artifact.from_dict(args)
            a.num_demos = 1
            a.save("not_suppose_to_save.json")

        self.assertEqual(
            str(e.exception),
            "Cannot save catalog artifacts that have changed since initialization. Detected differences in the following fields:\n - num_demos (changed)",
        )
