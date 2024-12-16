import json

from unitxt.artifact import (
    Artifact,
    MissingArtifactTypeError,
    UnrecognizedArtifactTypeError,
    json_dumps_with_artifacts,
    json_loads_with_artifacts,
)
from unitxt.card import TaskCard
from unitxt.logging_utils import get_logger
from unitxt.templates import InputOutputTemplate

from tests.utils import UnitxtTestCase

logger = get_logger()


class TestArtifactRecovery(UnitxtTestCase):
    def test_correct_artifact_recovery(self):
        args = {
            "__type__": "standard_recipe",
            "card": "cards.sst2",
            "template": {
                "__type__": "input_output_template",
                "input_format": "Given the following {type_of_input}, generate the corresponding {type_of_output}. {type_of_input}: {input}",
                "output_format": "{output}",
                "postprocessors": [
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                ],
            },
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        a = Artifact.from_dict(args)
        self.assertEqual(a.num_demos, 0)
        self.assertIsInstance(a.template, InputOutputTemplate)

    def test_correct_artifact_loading_with_json_loads(self):
        args = {
            "__type__": "standard_recipe",
            "card": "cards.sst2",
            "template": {
                "__type__": "input_output_template",
                "input_format": "Given the following {type_of_input}, generate the corresponding {type_of_output}. {type_of_input}: {input}",
                "output_format": "{output}",
                "postprocessors": [
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                ],
            },
            "demos_pool_size": 100,
            "num_demos": 0,
        }

        a = json_loads_with_artifacts(json.dumps(args))
        self.assertEqual(a.num_demos, 0)

        a = json_loads_with_artifacts(json.dumps({"x": args}))
        self.assertEqual(a["x"].num_demos, 0)

        self.assertIsInstance(a["x"].card, TaskCard)
        self.assertIsInstance(a["x"].template, InputOutputTemplate)

        d = json.loads(json_dumps_with_artifacts(a))
        self.assertDictEqual(d, {"x": args})

    def test_correct_artifact_recovery_with_overwrite(self):
        args = {
            "__type__": "standard_recipe",
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
            "__type__": "standard_recipe",
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
