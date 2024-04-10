from unitxt.api import evaluate, load_dataset, produce

from tests.utils import UnitxtTestCase


class TestAPI(UnitxtTestCase):
    def test_load_dataset(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        instance = {
            "metrics": ["metrics.spearman"],
            "source": "Given this sentence: 'A plane is taking off.', on a scale of 1 to 5, what is the similarity to this text An air plane is taking off.?\n",
            "target": "5.0",
            "references": ["5.0"],
            "task_data": '{"text1": "A plane is taking off.", "text2": "An air plane is taking off.", "attribute_name": "similarity", "min_value": "1", "max_value": "5", "attribute_value": 5.0}',
            "group": "unitxt",
            "postprocessors": ["processors.to_string_stripped"],
        }
        self.assertEqual(len(dataset["train"]), 5)
        self.assertDictEqual(dataset["train"][0], instance)

    def test_evaluate(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        predictions = ["2.5", "2.5", "2.2", "3", "4"]
        results = evaluate(predictions, dataset["train"])
        self.assertAlmostEqual(results[0]["score"]["global"]["score"], 0.026, 3)

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
            "source": "Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.premise: Steve follows Fred's example in everything. He influences him hugely., hypothesis: Steve influences him hugely.\nThe entailment class is entailment\n\npremise: The police arrested all of the gang members. They were trying to stop the drug trade in the neighborhood., hypothesis: The police were trying to stop the drug trade in the neighborhood.\nThe entailment class is not entailment\n\npremise: It works perfectly, hypothesis: It works!\nThe entailment class is ",
            "target": "?",
            "references": ["?"],
            "task_data": '{"text_a": "It works perfectly", "text_a_type": "premise", "text_b": "It works!", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "label": "?"}',
            "group": "unitxt",
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
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
            "card=cards.wnli,template=templates.classification.multi_class.relation.default,demos_pool_size=5,num_demos=2",
        )[0]

        target = {
            "metrics": ["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
            "source": "Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.premise: Steve follows Fred's example in everything. He influences him hugely., hypothesis: Steve influences him hugely.\nThe entailment class is entailment\n\npremise: The police arrested all of the gang members. They were trying to stop the drug trade in the neighborhood., hypothesis: The police were trying to stop the drug trade in the neighborhood.\nThe entailment class is not entailment\n\npremise: It works perfectly, hypothesis: It works!\nThe entailment class is ",
            "target": "?",
            "references": ["?"],
            "task_data": '{"text_a": "It works perfectly", "text_a_type": "premise", "text_b": "It works!", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "label": "?"}',
            "group": "unitxt",
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
        }

        self.assertDictEqual(target, result)
