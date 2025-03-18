from unitxt.artifact import (
    Artifact,
    MissingArtifactTypeError,
    UnrecognizedArtifactTypeError,
)
from unitxt.logging_utils import get_logger

from tests.utils import UnitxtTestCase

logger = get_logger()


class TestArtifactRecovery(UnitxtTestCase):
    def test_correct_artifact_recovery(self):
        args = {
            "__type__": "dataset_recipe",
            "card": "cards.sst2",
            "template_card_index": 0,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        a = Artifact.from_dict(args)
        self.assertEqual(a.num_demos, 0)

    def test_correct_artifact_recovery_with_overwrite(self):
        args = {
            "__type__": "dataset_recipe",
            "card": "cards.sst2",
            "template_card_index": 0,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        a = Artifact.from_dict(args, overwrite_args={"num_demos": 1})
        self.assertEqual(a.num_demos, 1)

    def test_bad_artifact_recovery_missing_type(self):
        args = {
            "card": "cards.sst2",
            "template_card_index": 1000,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        with self.assertRaises(MissingArtifactTypeError):
            Artifact.from_dict(args)

    def test_bad_artifact_recovery_bad_type(self):
        args = {
            "__type__": "dataset_recipe",
            "card": "cards.sst2",
            "template_card_index": 1000,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        with self.assertRaises(ValueError):
            Artifact.from_dict(args)

        try:
            Artifact.from_dict(args)
        except Exception as e:
            logger.info(e)

    def test_subclass_registration_and_loading(self):
        args = {
            "__type__": "dummy_not_exist",
        }
        with self.assertRaises(UnrecognizedArtifactTypeError):
            Artifact.from_dict(args)

        try:
            Artifact.from_dict(args)
        except UnrecognizedArtifactTypeError as e:
            logger.info("The error message (not a real error):", e)

        class DummyExistForLoading(Artifact):
            pass

        args = {
            "__type__": "dummy_exist_for_loading",
        }
        Artifact.from_dict(args)
