import unittest

from unitxt import produce
from unitxt.inference import HFLlavaInferenceEngine, HFPipelineBasedInferenceEngine
from unitxt.standard import StandardRecipe

from tests.utils import UnitxtTestCase


class TestInferenceEngine(UnitxtTestCase):
    def test_pipeline_based_inference_engine(self):
        inference_model = HFPipelineBasedInferenceEngine(
            model_name="google/flan-t5-small", max_new_tokens=32
        )
        assert inference_model._is_loaded()

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

    def test_pipeline_based_inference_engine_lzay_load(self):
        inference_model = HFPipelineBasedInferenceEngine(
            model_name="google/flan-t5-small", max_new_tokens=32, lazy_load=True
        )
        assert not inference_model._is_loaded()
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

    def test_dataset_verification_inference_engine(self):
        inference_model = HFPipelineBasedInferenceEngine(
            model_name="google/flan-t5-small",
            max_new_tokens=32,
            data_classification_policy=["public"],
        )
        dataset = [{"source": "", "data_classification_policy": ["pii"]}]
        with self.assertRaises(ValueError) as e:
            inference_model.infer(dataset)
        self.assertEqual(
            str(e.exception),
            f"The instance '{dataset[0]} 'has the following data classification policy "
            f"'{dataset[0]['data_classification_policy']}', however, the artifact "
            f"'{inference_model.get_pretty_print_name()}' is only configured to support the "
            f"data with classification '{inference_model.data_classification_policy}'. To "
            f"enable this either change the 'data_classification_policy' attribute of the "
            f"artifact, or modify the environment variable 'UNITXT_DATA_CLASSIFICATION_POLICY' "
            f"accordingly.",
        )

    def test_llava_inference_engine(self):
        import unitxt

        if not unitxt.settings.use_eager_execution:
            inference_model = HFLlavaInferenceEngine(
                model_name="llava-hf/llava-interleave-qwen-0.5b-hf", max_new_tokens=3
            )

            dataset = StandardRecipe(
                card="cards.doc_vqa.en",
                template="templates.qa.with_context.with_type",
                format="formats.models.llava_interleave",
                loader_limit=30,
            )()

            test_dataset = [dataset["test"].peek()]

            predictions = inference_model.infer(test_dataset)

            self.assertEqual(predictions[0], "The real image")


if __name__ == "__main__":
    unittest.main()
