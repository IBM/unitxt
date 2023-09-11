import unittest

from src.unitxt.artifact import Artifact, MissingArtifactType, UnrecognizedArtifactType


class TestArtifactRecovery(unittest.TestCase):
    def test_correct_artifact_recovery(self):
        args = {
            "type": "common_recipe",
            "card": "cards.sst2",
            "template_item": 0,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        artifact = Artifact.from_dict(args)

    def test_bad_artifact_recovery_missing_type(self):
        args = {
            # "type": "common_recipe",
            "card": "cards.sst2",
            "template_item": 1000,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        with self.assertRaises(MissingArtifactType):
            artifact = Artifact.from_dict(args)

    def test_bad_artifact_recovery_bad_type(self):
        args = {
            "type": "commmon_recipe",
            "card": "cards.sst2",
            "template_item": 1000,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        with self.assertRaises(UnrecognizedArtifactType):
            artifact = Artifact.from_dict(args)

        try:
            artifact = Artifact.from_dict(args)
        except Exception as e:
            print(e)
