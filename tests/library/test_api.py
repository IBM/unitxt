import json

import numpy as np
from unitxt.api import (
    create_dataset,
    evaluate,
    infer,
    load_dataset,
    post_process,
    produce,
)
from unitxt.card import TaskCard
from unitxt.loaders import LoadHF
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate, TemplatesList

from tests.utils import UnitxtTestCase, fillna, round_values


class TestAPI(UnitxtTestCase):
    maxDiff = None

    def test_load_dataset_without_template(self):
        dataset = load_dataset(
            "card=cards.stsb",
        )
        self.assertEqual(len(dataset["train"]), 282)

    def test_load_dataset(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5",
            use_cache=True,
        )
        target = {
            "metrics": ["metrics.spearman"],
            "data_classification_policy": ["public"],
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.cast_to_float_return_zero_if_failed",
            ],
            "target": "5.0",
            "references": ["5.0"],
            "source": "Given this sentence: 'A plane is taking off.', on a scale of 1.0 to 5.0, what is the similarity to this text 'An air plane is taking off.'?\n",
            "task_data": '{"text1": "A plane is taking off.", "text2": "An air plane is taking off.", "attribute_name": "similarity", "min_value": 1.0, "max_value": 5.0, "attribute_value": 5.0, "metadata": {"data_classification_policy": ["public"], "template": "templates.regression.two_texts.simple", "demos_pool_size": 0, "num_demos": 0}}',
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": [],
        }
        self.assertEqual(len(dataset["train"]), 5)
        result = dataset["train"][0]
        result_task_data = json.loads(result.pop("task_data"))
        target_task_data = json.loads(target.pop("task_data"))
        self.assertDictEqual(result, target)
        self.assertDictEqual(result_task_data, target_task_data)

    def test_load_dataset_with_multi_num_demos(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5,num_demos=[0,1],demos_pool_size=2,group_by=[num_demos,template]"
        )
        target = {
            "source": "Given this sentence: 'A man is spreading shreded cheese on a pizza.', on a scale of 1.0 to 5.0, what is the similarity to this text 'A man is spreading shredded cheese on an uncooked pizza.'?\n",
            "target": "3.8",
            "references": ["3.8"],
            "metrics": ["metrics.spearman"],
            "groups": [
                '{"num_demos": 0}',
                '{"template": "templates.regression.two_texts.simple"}',
            ],
            "subset": [],
            "media": {"images": [], "audios": []},
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.cast_to_float_return_zero_if_failed",
            ],
            "task_data": '{"text1": "A man is spreading shreded cheese on a pizza.", "text2": "A man is spreading shredded cheese on an uncooked pizza.", "attribute_name": "similarity", "min_value": 1.0, "max_value": 5.0, "metadata": {"data_classification_policy": ["public"], "demos_pool_size": 2, "num_demos": 0, "template": "templates.regression.two_texts.simple"}, "attribute_value": 3.799999952316284, "demos": []}',
            "data_classification_policy": ["public"],
        }
        self.assertEqual(len(dataset["train"]), 5)
        result = dataset["train"][0]
        result_task_data = json.loads(result.pop("task_data"))
        target_task_data = json.loads(target.pop("task_data"))
        self.assertDictEqual(result, target)
        self.assertDictEqual(result_task_data, target_task_data)

    def test_load_dataset_with_mixed_args(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,group_by=[num_demos,template]",
            max_train_instances=5,
            max_validation_instances=5,
            max_test_instances=5,
            num_demos=[0, 1],
            demos_pool_size=2,
        )
        target = {
            "source": "Given this sentence: 'A man is spreading shreded cheese on a pizza.', on a scale of 1.0 to 5.0, what is the similarity to this text 'A man is spreading shredded cheese on an uncooked pizza.'?\n",
            "target": "3.8",
            "references": ["3.8"],
            "metrics": ["metrics.spearman"],
            "groups": [
                '{"num_demos": 0}',
                '{"template": "templates.regression.two_texts.simple"}',
            ],
            "subset": [],
            "media": {"images": [], "audios": []},
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.cast_to_float_return_zero_if_failed",
            ],
            "task_data": '{"text1": "A man is spreading shreded cheese on a pizza.", "text2": "A man is spreading shredded cheese on an uncooked pizza.", "attribute_name": "similarity", "min_value": 1.0, "max_value": 5.0, "metadata": {"data_classification_policy": ["public"], "demos_pool_size": 2, "num_demos": 0, "template": "templates.regression.two_texts.simple"}, "attribute_value": 3.799999952316284, "demos": []}',
            "data_classification_policy": ["public"],
        }
        self.assertEqual(len(dataset["train"]), 5)
        result = dataset["train"][0]
        result_task_data = json.loads(result.pop("task_data"))
        target_task_data = json.loads(target.pop("task_data"))
        self.assertDictEqual(result, target)
        self.assertDictEqual(result_task_data, target_task_data)

    def test_load_dataset_with_multi_templates(self):
        dataset = load_dataset(
            "card=cards.stsb,template=[templates.regression.two_texts.simple,templates.key_val],max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        target = {
            "metrics": ["metrics.spearman"],
            "data_classification_policy": ["public"],
            "postprocessors": ["processors.to_string_stripped"],
            "target": "5.0",
            "references": ["5.0"],
            "source": "text1: A plane is taking off., text2: An air plane is taking off., attribute_name: similarity, min_value: 1.0, max_value: 5.0\n",
            "task_data": '{"text1": "A plane is taking off.", "text2": "An air plane is taking off.", "attribute_name": "similarity", "min_value": 1.0, "max_value": 5.0, "attribute_value": 5.0, "metadata": {"data_classification_policy": ["public"], "template": "templates.key_val", "demos_pool_size": 0, "num_demos": 0}}',
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": [],
        }

        self.assertEqual(len(dataset["train"]), 5)
        result = dataset["train"][0]
        result_task_data = json.loads(result.pop("task_data"))
        target_task_data = json.loads(target.pop("task_data"))
        self.assertDictEqual(result, target)
        self.assertDictEqual(result_task_data, target_task_data)

    def test_load_dataset_with_benchmark(self):
        dataset = load_dataset(
            "benchmarks.glue[max_samples_per_subset=1,loader_limit=300]"
        )
        first_result = dataset["test"].to_list()[0]
        last_result = dataset["test"].to_list()[-1]
        first_result_task_data = json.loads(first_result.pop("task_data"))
        last_result_task_data = json.loads(last_result.pop("task_data"))
        first_target = {
            "metrics": ["metrics.matthews_correlation"],
            "data_classification_policy": ["public"],
            "target": "acceptable",
            "references": ["acceptable"],
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
            "source": "Classify the grammatical acceptability of the following text to one of these options: unacceptable, acceptable.\ntext: The sailors rode the breeze clear of the rocks.\nThe grammatical acceptability is ",
            "task_data": '{"text": "The sailors rode the breeze clear of the rocks.", "text_type": "text", "classes": ["unacceptable", "acceptable"], "type_of_class": "grammatical acceptability", "label": "acceptable", "metadata": {"data_classification_policy": ["public"], "template": "templates.classification.multi_class.instruction", "demos_pool_size": 0, "num_demos": 0}}',
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": ["cola"],
        }
        last_target = {
            "metrics": ["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
            "data_classification_policy": ["public"],
            "target": "entailment",
            "references": ["entailment"],
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
            "source": "Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.\npremise: The drain is clogged with hair. It has to be cleaned.\nhypothesis: The hair has to be cleaned.\nThe entailment class is ",
            "task_data": '{"text_a": "The drain is clogged with hair. It has to be cleaned.", "text_a_type": "premise", "text_b": "The hair has to be cleaned.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "label": "entailment", "metadata": {"data_classification_policy": ["public"], "template": "templates.classification.multi_class.relation.default", "demos_pool_size": 0, "num_demos": 0}}',
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": ["wnli"],
        }
        first_target_task_data = json.loads(first_target.pop("task_data"))
        last_target_task_data = json.loads(last_target.pop("task_data"))
        self.assertDictEqual(first_result, first_target)
        self.assertDictEqual(first_result_task_data, first_target_task_data)
        self.assertDictEqual(last_result, last_target)
        self.assertDictEqual(last_result_task_data, last_target_task_data)

    def test_load_dataset_with_benchmark_mixed_args(self):
        dataset = load_dataset(
            "benchmarks.glue", max_samples_per_subset=1, loader_limit=300
        )
        first_result = dataset["test"].to_list()[0]
        last_result = dataset["test"].to_list()[-1]
        first_result_task_data = json.loads(first_result.pop("task_data"))
        last_result_task_data = json.loads(last_result.pop("task_data"))
        first_target = {
            "metrics": ["metrics.matthews_correlation"],
            "data_classification_policy": ["public"],
            "target": "acceptable",
            "references": ["acceptable"],
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
            "source": "Classify the grammatical acceptability of the following text to one of these options: unacceptable, acceptable.\ntext: The sailors rode the breeze clear of the rocks.\nThe grammatical acceptability is ",
            "task_data": '{"text": "The sailors rode the breeze clear of the rocks.", "text_type": "text", "classes": ["unacceptable", "acceptable"], "type_of_class": "grammatical acceptability", "label": "acceptable", "metadata": {"data_classification_policy": ["public"], "template": "templates.classification.multi_class.instruction", "demos_pool_size": 0, "num_demos": 0}}',
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": ["cola"],
        }
        last_target = {
            "metrics": ["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
            "data_classification_policy": ["public"],
            "target": "entailment",
            "references": ["entailment"],
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
            "source": "Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.\npremise: The drain is clogged with hair. It has to be cleaned.\nhypothesis: The hair has to be cleaned.\nThe entailment class is ",
            "task_data": '{"text_a": "The drain is clogged with hair. It has to be cleaned.", "text_a_type": "premise", "text_b": "The hair has to be cleaned.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "label": "entailment", "metadata": {"data_classification_policy": ["public"], "template": "templates.classification.multi_class.relation.default", "demos_pool_size": 0, "num_demos": 0}}',
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": ["wnli"],
        }
        first_target_task_data = json.loads(first_target.pop("task_data"))
        last_target_task_data = json.loads(last_target.pop("task_data"))
        self.assertDictEqual(first_result, first_target)
        self.assertDictEqual(first_result_task_data, first_target_task_data)
        self.assertDictEqual(last_result, last_target)
        self.assertDictEqual(last_result_task_data, last_target_task_data)

    def test_evaluate_with_group_by(self):
        dataset = load_dataset(
            "card=cards.stsb,template=[templates.regression.two_texts.simple,templates.regression.two_texts.title],max_train_instances=5,max_validation_instances=5,max_test_instances=5,group_by=[template]"
        )
        predictions = ["2.6", "2.5", "2.2", "3", "4"]
        results = evaluate(predictions, dataset["train"])
        self.assertDictEqual(
            round_values(fillna(results[0]["score"]["groups"], None), 3),
            {
                "template": {
                    "templates.regression.two_texts.title": {
                        "num_of_instances": 3,
                        "spearmanr": 0.5,
                        "score": 0.5,
                        "score_name": "spearmanr",
                        "score_ci_low": None,
                        "score_ci_high": None,
                        "spearmanr_ci_low": None,
                        "spearmanr_ci_high": None,
                        "spearmanr_p_value": 0.667,
                    },
                    "templates.regression.two_texts.simple": {
                        "num_of_instances": 2,
                        "spearmanr": -1.0,
                        "score": -1.0,
                        "score_name": "spearmanr",
                        "score_ci_low": -1.0,
                        "score_ci_high": -1.0,
                        "spearmanr_ci_low": -1.0,
                        "spearmanr_ci_high": -1.0,
                        "spearmanr_p_value": None,
                    },
                }
            },
        )

    def test_evaluate(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        predictions = ["2.45", "2.5", "2.2", "3", "4"]
        results = evaluate(predictions, dataset["train"])
        # Processors are not serialized by correctly yet
        instance_with_results = {
            "metrics": ["metrics.spearman"],
            "data_classification_policy": ["public"],
            "target": "5.0",
            "references": ["5.0"],
            "source": "Given this sentence: 'A plane is taking off.', on a scale of 1.0 to 5.0, what is the similarity to this text 'An air plane is taking off.'?\n",
            "task_data": {
                "text1": "A plane is taking off.",
                "text2": "An air plane is taking off.",
                "attribute_name": "similarity",
                "min_value": 1.0,
                "max_value": 5.0,
                "attribute_value": 5.0,
                "metadata": {
                    "data_classification_policy": ["public"],
                    "template": "templates.regression.two_texts.simple",
                    "demos_pool_size": 0,
                    "num_demos": 0,
                },
                "source": "Given this sentence: 'A plane is taking off.', on a scale of 1.0 to 5.0, what is the similarity to this text 'An air plane is taking off.'?\n",
            },
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": [],
            "prediction": "2.45",
            "processed_prediction": 2.45,
            "processed_references": [5.0],
            "score": {
                "global": {
                    "num_of_instances": 5,
                    "spearmanr": -0.10259783520851541,
                    "score": -0.10259783520851541,
                    "spearmanr_p_value": 0.8695979205185651,
                    "spearmanr_ci_low": -1.0,
                    "spearmanr_ci_high": 0.9946088287837848,
                    "score_ci_low": -1.0,
                    "score_ci_high": 0.9946088287837848,
                    "score_name": "spearmanr",
                },
                "instance": {
                    "score": np.nan,
                    "score_name": "spearmanr",
                    "spearmanr": np.nan,
                },
            },
        }
        del results[0]["postprocessors"]
        self.assertDictEqual(
            fillna(results[0], None), fillna(instance_with_results, None)
        )

    def test_evaluate_no_confidence_internal(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        predictions = ["2.5", "2.5", "2.2", "3", "4"]

        results = evaluate(
            predictions, dataset["train"], calc_confidence_intervals=False
        )

        instance_global_scores = {
            "num_of_instances": 5,
            "spearmanr": 0.026315789473684213,
            "score": 0.026315789473684213,
            "spearmanr_p_value": 0.9664975638949244,
            "score_name": "spearmanr",
        }

        self.assertDictEqual(
            fillna(results[0]["score"]["global"], None),
            fillna(instance_global_scores, None),
        )

    def test_evaluate_with_groups(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        predictions = ["2.4", "2.5", "2.2", "3", "4"]
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
                "metadata": {
                    "data_classification_policy": ["public"],
                    "template": "templates.regression.two_texts.simple",
                    "demos_pool_size": 0,
                    "num_demos": 0,
                },
                "source": "Given this sentence: 'A plane is taking off.', on a scale of 1.0 to 5.0, what is the similarity to this text 'An air plane is taking off.'?\n",
            },
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.cast_to_float_return_zero_if_failed",
            ],
            "data_classification_policy": ["public"],
            "prediction": "2.4",
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": [],
            "processed_prediction": 2.4,
            "processed_references": [5.0],
            "score": {
                "global": {
                    "num_of_instances": 5,
                    "score": -0.10259783520851541,
                    "score_ci_high": 0.9946088287837848,
                    "score_ci_low": -1.0,
                    "score_name": "spearmanr",
                    "spearmanr": -0.10259783520851541,
                    "spearmanr_ci_high": 0.9946088287837848,
                    "spearmanr_ci_low": -1.0,
                    "spearmanr_p_value": 0.8695979205185651,
                },
                "instance": {
                    "score": np.nan,
                    "score_name": "spearmanr",
                    "spearmanr": np.nan,
                },
            },
        }
        # Processors are not serialized by correctly yet
        del results[0]["postprocessors"]
        del instance_with_results["postprocessors"]
        self.assertDictEqual(results[0], instance_with_results)

    def test_post_process(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        predictions = ["2.5", "2.5", "2.2", "3", "4"]
        targets = [2.5, 2.5, 2.2, 3.0, 4.0]
        results = post_process(predictions, dataset["train"])
        self.assertListEqual(results, targets)

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
            "data_classification_policy": [],
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
            "source": "Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.\npremise: When Tatyana reached the cabin, her mother was sleeping. She was careful not to disturb her, undressing and climbing back into her berth.\nhypothesis: mother was careful not to disturb her, undressing and climbing back into her berth.\nThe entailment class is entailment\n\npremise: Steve follows Fred's example in everything. He influences him hugely.\nhypothesis: Steve influences him hugely.\nThe entailment class is entailment\n\npremise: It works perfectly\nhypothesis: It works!\nThe entailment class is ",
            "task_data": '{"text_a": "It works perfectly", "text_a_type": "premise", "text_b": "It works!", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": [], "num_demos": 2, "demos_pool_size": 5, "template": "templates.classification.multi_class.relation.default"}, "demos": [{"text_a": "When Tatyana reached the cabin, her mother was sleeping. She was careful not to disturb her, undressing and climbing back into her berth.", "text_a_type": "premise", "text_b": "mother was careful not to disturb her, undressing and climbing back into her berth.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": ["public"]}, "label": "entailment"}, {"text_a": "Steve follows Fred\'s example in everything. He influences him hugely.", "text_a_type": "premise", "text_b": "Steve influences him hugely.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": ["public"]}, "label": "entailment"}]}',
            "groups": [],
            "subset": [],
            "media": {"images": [], "audios": []},
        }

        self.assertDictEqual(target, result)

    def test_produce_with_task(self):
        result = produce(
            {
                "text_a": "It works perfectly",
                "text_b": "It works!",
                "classes": ["entailment", "not entailment"],
                "type_of_relation": "entailment",
                "text_a_type": "premise",
                "text_b_type": "hypothesis",
            },
            "task=tasks.classification.multi_class.relation,template=templates.classification.multi_class.relation.default",
        )

        target = {
            "metrics": ["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
            "data_classification_policy": [],
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
            "source": "Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.\npremise: It works perfectly\nhypothesis: It works!\nThe entailment class is ",
            "task_data": '{"text_a": "It works perfectly", "text_a_type": "premise", "text_b": "It works!", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": [], "num_demos": 0, "demos_pool_size": 0, "template": "templates.classification.multi_class.relation.default"}}',
            "groups": [],
            "subset": [],
            "media": {"images": [], "audios": []},
        }

        self.assertDictEqual(target, result)

    def test_produce_with_recipe_with_list_of_instances(self):
        result = produce(
            [
                {
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
            "data_classification_policy": [],
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
            "source": "Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.\npremise: When Tatyana reached the cabin, her mother was sleeping. She was careful not to disturb her, undressing and climbing back into her berth.\nhypothesis: mother was careful not to disturb her, undressing and climbing back into her berth.\nThe entailment class is entailment\n\npremise: Steve follows Fred's example in everything. He influences him hugely.\nhypothesis: Steve influences him hugely.\nThe entailment class is entailment\n\npremise: It works perfectly\nhypothesis: It works!\nThe entailment class is ",
            "task_data": '{"text_a": "It works perfectly", "text_a_type": "premise", "text_b": "It works!", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": [], "num_demos": 2, "demos_pool_size": 5, "template": "templates.classification.multi_class.relation.default"}, "demos": [{"text_a": "When Tatyana reached the cabin, her mother was sleeping. She was careful not to disturb her, undressing and climbing back into her berth.", "text_a_type": "premise", "text_b": "mother was careful not to disturb her, undressing and climbing back into her berth.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": ["public"]}, "label": "entailment"}, {"text_a": "Steve follows Fred\'s example in everything. He influences him hugely.", "text_a_type": "premise", "text_b": "Steve influences him hugely.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": ["public"]}, "label": "entailment"}]}',
            "groups": [],
            "subset": [],
            "media": {"images": [], "audios": []},
        }

        self.assertDictEqual(target, result)

    def test_load_dataset_from_dict(self):
        card = TaskCard(
            loader=LoadHF(path="glue", name="wnli"),
            task=Task(
                input_fields=["sentence1", "sentence2"],
                reference_fields=["label"],
                metrics=["metrics.accuracy"],
            ),
            templates=TemplatesList(
                [
                    InputOutputTemplate(
                        input_format="Sentence1: {sentence1} Sentence2: {sentence2}",
                        output_format="{label}",
                    ),
                    InputOutputTemplate(
                        input_format="Sentence2: {sentence2} Sentence1: {sentence1}",
                        output_format="{label}",
                    ),
                ]
            ),
        )

        dataset = load_dataset(card=card, template_card_index=1, loader_limit=5)

        self.assertEqual(len(dataset["train"]), 5)
        self.assertEqual(
            dataset["train"]["source"][0].strip(),
            "Sentence2: The carrot had a hole. "
            "Sentence1: I stuck a pin through a carrot. "
            "When I pulled the pin out, it had a hole.",
        )
        self.assertEqual(dataset["train"]["metrics"][0], ["metrics.accuracy"])

    def test_infer(self):
        engine = "engines.model.flan.t5_small.hf"
        recipe = "card=cards.almost_evil,template=templates.qa.open.simple,demos_pool_size=0,num_demos=0"
        instances = [
            {"question": "How many days there are in a week"},
            {
                "question": "If a ate an apple in the morning, and one in the evening, how many apples did I eat?",
            },
        ]
        predictions = infer(instances, engine, recipe)
        targets = [
            "Question: How many days there are in a week\nAnswer:",
            "Question: If a ate an apple in the morning, and one in the evening, how many apples did I eat?\nAnswer:",
        ]
        self.assertListEqual(predictions, targets)

    def test_infer_with_task(self):
        engine = "engines.model.flan.t5_small.hf"
        recipe = "task=tasks.qa.open,template=templates.qa.open.simple"
        instances = [
            {"question": "How many days there are in a week"},
            {
                "question": "If a ate an apple in the morning, and one in the evening, how many apples did I eat?",
            },
        ]
        predictions = infer(instances, engine, recipe)
        targets = [
            "Question: How many days there are in a week\nAnswer:",
            "Question: If a ate an apple in the morning, and one in the evening, how many apples did I eat?\nAnswer:",
        ]
        self.assertListEqual(predictions, targets)

    def test_create_dataset_with_non_seralizble_object(self):
        import numpy as np
        from PIL import Image

        random_image = Image.fromarray(
            np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        )

        instances = [
            {
                "context": {"image": random_image, "format": "JPEG"},
                "context_type": "image",
                "question": "What is the capital of Texas?",
                "answers": ["Austin"],
            },
            {
                "context": {"image": random_image, "format": "JPEG"},
                "context_type": "image",
                "question": "What is the color of the sky?",
                "answers": ["Blue"],
            },
        ]

        create_dataset(
            task="tasks.qa.with_context",
            format="formats.chat_api",
            test_set=instances,
        )
        create_dataset(
            task="tasks.qa.with_context",
            format="formats.chat_api",
            test_set=instances,
            train_set=instances,
            validation_set=instances,
        )
