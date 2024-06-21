import numpy as np
from unitxt.api import evaluate, load_dataset, produce

from tests.utils import UnitxtTestCase


class TestAPI(UnitxtTestCase):
    def test_load_dataset(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        instance = {
            "metrics": ["metrics.spearman"],
            "source": "Given this sentence: 'A plane is taking off.', on a scale of 1.0 to 5.0, what is the similarity to this text 'An air plane is taking off.'?\n",
            "target": "5.0",
            "references": ["5.0"],
            "task_data": '{"text1": "A plane is taking off.", '
            '"text2": "An air plane is taking off.", '
            '"attribute_name": "similarity", '
            '"min_value": 1.0, '
            '"max_value": 5.0, '
            '"attribute_value": 5.0, '
            '"metadata": {"template": "templates.regression.two_texts.simple"}}',
            "group": "unitxt",
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.cast_to_float_return_zero_if_failed",
            ],
            "data_classification_policy": ["public"],
        }
        self.assertEqual(len(dataset["train"]), 5)
        self.assertDictEqual(dataset["train"][0], instance)

    def test_evaluate(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        predictions = ["2.5", "2.5", "2.2", "3", "4"]
        results = evaluate(predictions, dataset["train"])
        instance_with_results = {
            "metrics": ["metrics.spearman"],
            "source": "Given this sentence: 'A plane is taking off.', on a scale of 1.0 to 5.0, what is the similarity to this text 'An air plane is taking off.'?\n",
            "target": "5.0",
            "references": ["5.0"],
            "task_data": {
                "text1": "A plane is taking off.",
                "text2": "An air plane is taking off.",
                "attribute_name": "similarity",
                "min_value": 1.0,
                "max_value": 5.0,
                "attribute_value": 5.0,
                "metadata": {"template": "templates.regression.two_texts.simple"},
            },
            "group": "unitxt",
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.cast_to_float_return_zero_if_failed",
            ],
            "data_classification_policy": ["public"],
            "prediction": "2.5",
            "processed_prediction": 2.5,
            "processed_references": [5],
            "score": {
                "global": {
                    "score": 0.026315789473684213,
                    "score_ci_high": 0.9639697714358006,
                    "score_ci_low": -0.970678676196682,
                    "score_name": "spearmanr",
                    "spearmanr": 0.026315789473684213,
                    "spearmanr_ci_high": 0.9639697714358006,
                    "spearmanr_ci_low": -0.970678676196682,
                },
                "instance": {
                    "score": np.nan,
                    "score_name": "spearmanr",
                    "spearmanr": np.nan,
                },
            },
        }
        self.assertDictEqual(results[0], instance_with_results)

    def test_evaluate_with_metrics_external_setup(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5,metrics=[metrics.accuracy],postprocessors=[processors.first_character]"
        )
        self.assertEqual(dataset["train"][0]["metrics"], ["metrics.accuracy"])
        self.assertEqual(
            dataset["train"][0]["postprocessors"], ["processors.first_character"]
        )
        predictions = ["2.5", "2.5", "2.2", "3", "4"]
        results = evaluate(predictions, dataset["train"])
        self.assertAlmostEqual(results[0]["score"]["global"]["score"], 0.2, 3)

    def test_produce_with_recipe(self):
        result = produce(
            {
                "label": "?",
                "text_a": "It works perfectly",
                "text_b": "It works!",
                "classes": ["entailment", "not entailment"],
                "type_of_relation": "entailment",
                "text_a_type": "premise",
                "text_b_type": "hypothesis",
            },
            "card=cards.wnli,template=templates.classification.multi_class.relation.default,demos_pool_size=5,num_demos=2",
        )

        target = {
            "metrics": ["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
            "source": "Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.\npremise: Steve follows Fred's example in everything. He influences him hugely.\nhypothesis: Steve influences him hugely.\nThe entailment class is entailment\n\npremise: The police arrested all of the gang members. They were trying to stop the drug trade in the neighborhood.\nhypothesis: The police were trying to stop the drug trade in the neighborhood.\nThe entailment class is not entailment\n\npremise: It works perfectly\nhypothesis: It works!\nThe entailment class is ",
            "target": "?",
            "references": ["?"],
            "task_data": '{"text_a": "It works perfectly", '
            '"text_a_type": "premise", '
            '"text_b": "It works!", '
            '"text_b_type": "hypothesis", '
            '"classes": ["entailment", "not entailment"], '
            '"type_of_relation": "entailment", '
            '"label": "?", '
            '"metadata": {"template": "templates.classification.multi_class.relation.default"}}',
            "group": "unitxt",
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
            "data_classification_policy": [],
        }

        self.assertDictEqual(target, result)

    def test_produce_with_recipe_with_list_of_instances(self):
        result = produce(
            [
                {
                    "label": "?",
                    "text_a": "It works perfectly",
                    "text_b": "It works!",
                    "classes": ["entailment", "not entailment"],
                    "type_of_relation": "entailment",
                    "text_a_type": "premise",
                    "text_b_type": "hypothesis",
                }
            ],
            "card=cards.wnli,template=templates.classification.multi_class.relation.default,demos_pool_size=5,num_demos=2,loader_limit=10",
        )[0]

        target = {
            "metrics": ["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
            "source": "Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.\npremise: Steve follows Fred's example in everything. He influences him hugely.\nhypothesis: Steve influences him hugely.\nThe entailment class is entailment\n\npremise: The police arrested all of the gang members. They were trying to stop the drug trade in the neighborhood.\nhypothesis: The police were trying to stop the drug trade in the neighborhood.\nThe entailment class is not entailment\n\npremise: It works perfectly\nhypothesis: It works!\nThe entailment class is ",
            "target": "?",
            "references": ["?"],
            "task_data": '{"text_a": "It works perfectly", '
            '"text_a_type": "premise", '
            '"text_b": "It works!", '
            '"text_b_type": "hypothesis", '
            '"classes": ["entailment", "not entailment"], '
            '"type_of_relation": "entailment", '
            '"label": "?", '
            '"metadata": {"template": "templates.classification.multi_class.relation.default"}}',
            "group": "unitxt",
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
            "data_classification_policy": [],
        }

        self.assertDictEqual(target, result)
