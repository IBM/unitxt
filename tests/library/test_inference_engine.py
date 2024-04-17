import unittest

from unitxt import produce
from unitxt.inference import HFPipelineBasedInferenceEngine

from tests.utils import UnitxtTestCase


class TestInferenceEngine(UnitxtTestCase):
    def test_pipeline_based_inference_engine(self):
        inference_model = HFPipelineBasedInferenceEngine(
            model_name="google/flan-t5-small", max_new_tokens=32
        )
        recipe = "card=cards.almost_evil,template=templates.qa.open.simple,demos_pool_size=0,num_demos=0"
        instances = [
            {"question": "How many days there are in a week", "answers": ["7"]},
            {
                "question": "If a ate an apple in the morning, and one in the evening, how many apples did I eat?",
                "answers": ["2"],
            },
        ]
        dataset = produce(instances, recipe)

        predictions = inference_model.infer(dataset)

        targets = ["365", "1"]
        self.assertListEqual(predictions, targets)


if __name__ == "__main__":
    unittest.main()
