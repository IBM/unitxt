from unitxt.artifact import (
    Artifact,
    MissingArtifactTypeError,
    UnrecognizedArtifactTypeError,
    from_dict,
)
from unitxt.logging_utils import get_logger

from tests.utils import UnitxtTestCase

logger = get_logger()


class TestArtifactRecovery(UnitxtTestCase):
    def test_correct_artifact_recovery(self):
        args = {
            "__type__": {"module": "unitxt.standard", "name": "DatasetRecipe"},
            "card": "cards.sst2",
            "template_card_index": 0,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        a = from_dict(args)
        self.assertEqual(a.num_demos, 0)

    def test_correct_artifact_recovery_with_overwrite(self):
        args = {
            "__type__": {"module": "unitxt.standard", "name": "DatasetRecipe"},
            "card": "cards.sst2",
            "template_card_index": 0,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        a = from_dict(args, overwrite_args={"num_demos": 1})
        self.assertEqual(a.num_demos, 1)

    def test_bad_artifact_recovery_missing_type(self):
        args = {
            "card": "cards.sst2",
            "template_card_index": 1000,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        with self.assertRaises(MissingArtifactTypeError):
            from_dict(args)

    def test_bad_artifact_recovery_bad_type(self):
        args = {
            "__type__": {"module": "unitxt.standard", "name": "DatasetRecipe"},
            "card": "cards.sst2",
            "template_card_index": 1000,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        with self.assertRaises(ValueError):
            from_dict(args)

        try:
            from_dict(args)
        except Exception as e:
            logger.info(e)

    def test_subclass_registration_and_loading(self):
        args = {
            "__type__": {"module": "dummy_not_exist", "name": "Nowhere"},
        }
        with self.assertRaises(UnrecognizedArtifactTypeError):
            from_dict(args)

        try:
            from_dict(args)
        except UnrecognizedArtifactTypeError as e:
            logger.info("The error message (not a real error):", e)

        class DummyExistsForLoading(Artifact):
            pass

        args = {
            "__type__": {"module": "class_register", "name": "DummyExistsForLoading"},
        }

        DummyExistsForLoading()

        artifact = from_dict(args)
        self.assertEqual(DummyExistsForLoading, artifact.__class__)

        Artifact._class_register.pop("DummyExistsForLoading")
        with self.assertRaises(ValueError):
            artifact = from_dict(args)
