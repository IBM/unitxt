import collections
import copy
import json
import re
from typing import Any, Dict

from unitxt import dataset_file
from unitxt.artifact import fetch_artifact
from unitxt.card import TaskCard
from unitxt.formats import SystemFormat
from unitxt.loaders import LoadFromDictionary
from unitxt.serializers import SingleTypeSerializer, TableSerializer
from unitxt.standard import StandardRecipe, StandardRecipeWithIndexes
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate, TemplatesList
from unitxt.text_utils import print_dict
from unitxt.types import Table

from tests.utils import UnitxtTestCase


class TestRecipes(UnitxtTestCase):
    def test_standard_recipe(self):
        recipe = StandardRecipe(
            card="cards.wnli",
            template=InputOutputTemplate(
                input_format="{text_a}",
                output_format="{label}",
                instruction="classify",
            ),
            format=SystemFormat(
                demo_format="User:{source}\nAgent:{target}\n\n",
                model_input_format="{instruction}\n\n{demos}User:{source}\nAgent:",
            ),
        )
        stream = recipe()

        for instance in stream["train"]:
            print_dict(instance)
            del instance["task_data"]
            self.assertDictEqual(
                instance,
                {
                    "metrics": [
                        "metrics.f1_micro",
                        "metrics.accuracy",
                        "metrics.f1_macro",
                    ],
                    "source": "classify\n\nUser:I stuck a pin through a carrot. When I pulled the pin out, it had a hole.\nAgent:",
                    "target": "not entailment",
                    "references": ["not entailment"],
                    "groups": [],
                    "media": {"audios": [], "images": []},
                    "subset": [],
                    "postprocessors": ["processors.to_string_stripped"],
                    "data_classification_policy": ["public"],
                },
            )
            break

    def test_standard_recipe_with_catalog(self):
        recipe = StandardRecipe(
            card="cards.mmlu.marketing",
            system_prompt="system_prompts.models.llama",
            template="templates.qa.multiple_choice.with_topic.lm_eval_harness",
            format="formats.user_agent",
            demos_pool_size=5,
            num_demos=3,
        )

        stream = recipe()

        for instance in stream["test"]:
            print_dict(instance)
            break

    def test_standard_recipe_production_without_demos(self):
        recipe = StandardRecipe(
            card="cards.mmlu.marketing",
            system_prompt="system_prompts.models.llama",
            template="templates.qa.multiple_choice.with_topic.lm_eval_harness",
            format="formats.user_agent",
        )

        result = recipe.produce(
            [
                {
                    "question": "what?",
                    "choices": ["yes", "not", "maybe"],
                    "topic": "testing",
                }
            ]
        )[0]

        target = {
            "metrics": ["metrics.accuracy"],
            "data_classification_policy": [],
            "postprocessors": ["processors.first_character"],
            "source": "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n\n\n\n\nUser:The following are multiple choice questions (with answers) about testing.\n\nwhat?\nA. yes\nB. not\nC. maybe\nAnswer:\nAgent:",
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": [],
        }

        del result["task_data"]

        self.assertDictEqual(result, target)

    def test_standard_recipe_production_consistency(self):
        recipe = StandardRecipe(
            card="cards.mmlu.marketing",
            system_prompt="system_prompts.models.llama",
            template="templates.qa.multiple_choice.with_topic.lm_eval_harness",
            format="formats.user_agent",
            demos_pool_size=5,
            num_demos=1,
        )

        instances = [
            {
                "question": "what?",
                "choices": ["yes", "not", "maybe"],
                "answer": "maybe",
                "topic": "testing",
            }
        ]

        self.assertListEqual(
            recipe.production_demos_pool(), recipe.production_demos_pool()
        )

        self.assertDictEqual(
            recipe.produce(instances)[0],
            recipe.produce(instances)[0],
        )

        i1 = recipe.production_preprocess(instances)[0]
        i2 = recipe.production_preprocess(instances)[0]
        for meta_data in ["card", "template", "format", "system_prompt"]:
            if meta_data in i1["recipe_metadata"]:
                i1["recipe_metadata"][meta_data] = i1["recipe_metadata"][
                    meta_data
                ]._to_raw_dict()
                if not isinstance(i2["recipe_metadata"][meta_data], dict):
                    i2["recipe_metadata"][meta_data] = i2["recipe_metadata"][
                        meta_data
                    ]._to_raw_dict()

        self.assertDictEqual(i1, i2)

    def test_standard_recipe_production_with_demos(self):
        recipe = StandardRecipe(
            card="cards.mmlu.marketing",
            system_prompt="system_prompts.models.llama",
            template="templates.qa.multiple_choice.with_topic.lm_eval_harness",
            format="formats.user_agent",
            demos_pool_size=5,
            num_demos=3,
        )

        result = recipe.produce(
            [
                {
                    "question": "what?",
                    "choices": ["yes", "not", "maybe"],
                    "topic": "testing",
                }
            ]
        )[0]

        target = {
            "metrics": ["metrics.accuracy"],
            "data_classification_policy": [],
            "postprocessors": ["processors.first_character"],
            "source": "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n\n\n\nUser: The following are multiple choice questions (with answers) about marketing.\n\nAlthough the content and quality can be as controlled as direct mail, response rates of this medium are lower because of the lack of a personal address mechanism. This media format is known as:\nA. Care lines.\nB. Direct mail.\nC. Inserts.\nD. Door to door.\nAnswer:\nAgent:  D\n\nUser: The following are multiple choice questions (with answers) about marketing.\n\n _____________ is a natural outcome when combining demographic and geographic variables.\nA. Geodemographics\nB. Product differentiation.\nC. ANSOFF matrix.\nD. Brand management.\nAnswer:\nAgent:  A\n\nUser: The following are multiple choice questions (with answers) about marketing.\n\nIn an organization, the group of people tasked with buying decisions is referred to as the _______________.\nA. Outsourcing unit.\nB. Procurement centre.\nC. Chief executive unit.\nD. Decision-making unit.\nAnswer:\nAgent:  D\n\n\nUser:The following are multiple choice questions (with answers) about testing.\n\nwhat?\nA. yes\nB. not\nC. maybe\nAnswer:\nAgent:",
            "task_data": '{"topic": "testing", "question": "what?", "choices": ["yes", "not", "maybe"], "options": [" A", " B", " C"], "metadata": {"data_classification_policy": [], "num_demos": 3, "template": "templates.qa.multiple_choice.with_topic.lm_eval_harness"}, "demos": [{"topic": "marketing", "question": "Although the content and quality can be as controlled as direct mail, response rates of this medium are lower because of the lack of a personal address mechanism. This media format is known as:", "choices": ["Care lines.", "Direct mail.", "Inserts.", "Door to door."], "options": [" A", " B", " C", " D"], "metadata": {"data_classification_policy": ["public"]}, "answer": 3}, {"topic": "marketing", "question": " _____________ is a natural outcome when combining demographic and geographic variables.", "choices": ["Geodemographics", "Product differentiation.", "ANSOFF matrix.", "Brand management."], "options": [" A", " B", " C", " D"], "metadata": {"data_classification_policy": ["public"]}, "answer": 0}, {"topic": "marketing", "question": "In an organization, the group of people tasked with buying decisions is referred to as the _______________.", "choices": ["Outsourcing unit.", "Procurement centre.", "Chief executive unit.", "Decision-making unit."], "options": [" A", " B", " C", " D"], "metadata": {"data_classification_policy": ["public"]}, "answer": 3}]}',
            "groups": [],
            "subset": [],
            "media": {"images": [], "audios": []},
        }

        target_task_data = json.loads(target.pop("task_data"))
        result_task_data = json.loads(result.pop("task_data"))

        self.assertDictEqual(result, target)
        self.assertDictEqual(target_task_data, result_task_data)

    def test_standard_recipe_with_indexes_with_catalog(self):
        recipe = StandardRecipe(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template_card_index=0,
            format="formats.user_agent",
            demos_pool_size=100,
            num_demos=3,
        )

        stream = recipe()

        for instance in stream["train"]:
            print_dict(instance)
            break

    def test_standard_recipe_with_demos_not_removed_from_data(self):
        recipe = StandardRecipe(
            card="cards.wnli",
            template_card_index=0,
            demos_pool_size=100,
            num_demos=3,
            demos_removed_from_data=True,
        )

        stream = recipe()
        n_trains_remove_demos = len(list(stream["train"]))
        n_demos_remove_demos = len(list(stream["demos_pool"]))

        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            template_card_index=0,
            demos_pool_size=100,
            num_demos=3,
            demos_removed_from_data=False,
        )

        stream = recipe()
        n_trains_keep_demos = len(list(stream["train"]))
        n_demos_keep_demos = len(list(stream["demos_pool"]))

        self.assertEqual(
            n_trains_keep_demos, n_trains_remove_demos + n_demos_remove_demos
        )
        self.assertEqual(n_demos_keep_demos, n_demos_remove_demos)

    def test_empty_template(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template="templates.empty",
            format="formats.user_agent",
            demos_pool_size=100,
            num_demos=3,
        )

        target = {
            "metrics": ["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
            "data_classification_policy": ["public"],
            "media": {"images": [], "audios": []},
            "postprocessors": ["processors.to_string_stripped"],
            "target": "not entailment",
            "references": ["not entailment"],
            "source": "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n\n\n\nUser: Emma did not pass the ball to Janie although she was open., premise, She saw that Janie was open., hypothesis, entailment, not entailment, entailment\nAgent: not entailment\n\nUser: The foxes are getting in at night and attacking the chickens. I shall have to kill them., premise, I shall have to kill The foxes., hypothesis, entailment, not entailment, entailment\nAgent: not entailment\n\nUser: Fred is the only man alive who still remembers my father as an infant. When Fred first saw my father, he was twelve years old., premise, When Fred first saw my father, My father was twelve years old., hypothesis, entailment, not entailment, entailment\nAgent: entailment\n\n\nUser:Grace was happy to trade me her sweater for my jacket. She thinks it looks dowdy on her., premise, The sweater looks dowdy on her., hypothesis, entailment, not entailment, entailment\nAgent:",
            "task_data": '{"text_a": "Grace was happy to trade me her sweater for my jacket. She thinks it looks dowdy on her.", "text_a_type": "premise", "text_b": "The sweater looks dowdy on her.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": ["public"], "num_demos": 3, "template": "templates.empty"}, "label": "not entailment", "demos": [{"text_a": "Emma did not pass the ball to Janie although she was open.", "text_a_type": "premise", "text_b": "She saw that Janie was open.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": ["public"]}, "label": "not entailment"}, {"text_a": "The foxes are getting in at night and attacking the chickens. I shall have to kill them.", "text_a_type": "premise", "text_b": "I shall have to kill The foxes.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": ["public"]}, "label": "not entailment"}, {"text_a": "Fred is the only man alive who still remembers my father as an infant. When Fred first saw my father, he was twelve years old.", "text_a_type": "premise", "text_b": "When Fred first saw my father, My father was twelve years old.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": ["public"]}, "label": "entailment"}]}',
            "groups": [],
            "subset": [],
        }

        stream = recipe()
        result = stream["train"].peek()

        target_task_data = json.loads(target.pop("task_data"))
        result_task_data = json.loads(result.pop("task_data"))

        self.assertDictEqual(result, target)
        self.assertDictEqual(target_task_data, result_task_data)

    def test_key_val_template(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            demos_pool_size=100,
            num_demos=3,
        )

        target = {
            "metrics": ["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
            "data_classification_policy": ["public"],
            "target": "not entailment",
            "references": ["not entailment"],
            "postprocessors": ["processors.to_string_stripped"],
            "source": "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n\n\n\nUser: text_a: Emma did not pass the ball to Janie although she was open., text_a_type: premise, text_b: She saw that Janie was open., text_b_type: hypothesis, classes: entailment, not entailment, type_of_relation: entailment\nAgent: not entailment\n\nUser: text_a: The foxes are getting in at night and attacking the chickens. I shall have to kill them., text_a_type: premise, text_b: I shall have to kill The foxes., text_b_type: hypothesis, classes: entailment, not entailment, type_of_relation: entailment\nAgent: not entailment\n\nUser: text_a: Fred is the only man alive who still remembers my father as an infant. When Fred first saw my father, he was twelve years old., text_a_type: premise, text_b: When Fred first saw my father, My father was twelve years old., text_b_type: hypothesis, classes: entailment, not entailment, type_of_relation: entailment\nAgent: entailment\n\n\nUser:text_a: Grace was happy to trade me her sweater for my jacket. She thinks it looks dowdy on her., text_a_type: premise, text_b: The sweater looks dowdy on her., text_b_type: hypothesis, classes: entailment, not entailment, type_of_relation: entailment\nAgent:",
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": [],
        }

        target_task_data = {
            "text_a": "Grace was happy to trade me her sweater for my jacket. She thinks it looks dowdy on her.",
            "text_a_type": "premise",
            "text_b": "The sweater looks dowdy on her.",
            "text_b_type": "hypothesis",
            "classes": ["entailment", "not entailment"],
            "type_of_relation": "entailment",
            "metadata": {
                "data_classification_policy": ["public"],
                "num_demos": 3,
                "template": "templates.key_val",
            },
            "label": "not entailment",
            "demos": [
                {
                    "text_a": "Emma did not pass the ball to Janie although she was open.",
                    "text_a_type": "premise",
                    "text_b": "She saw that Janie was open.",
                    "text_b_type": "hypothesis",
                    "classes": ["entailment", "not entailment"],
                    "type_of_relation": "entailment",
                    "metadata": {"data_classification_policy": ["public"]},
                    "label": "not entailment",
                },
                {
                    "text_a": "The foxes are getting in at night and attacking the chickens. I shall have to kill them.",
                    "text_a_type": "premise",
                    "text_b": "I shall have to kill The foxes.",
                    "text_b_type": "hypothesis",
                    "classes": ["entailment", "not entailment"],
                    "type_of_relation": "entailment",
                    "metadata": {"data_classification_policy": ["public"]},
                    "label": "not entailment",
                },
                {
                    "text_a": "Fred is the only man alive who still remembers my father as an infant. When Fred first saw my father, he was twelve years old.",
                    "text_a_type": "premise",
                    "text_b": "When Fred first saw my father, My father was twelve years old.",
                    "text_b_type": "hypothesis",
                    "classes": ["entailment", "not entailment"],
                    "type_of_relation": "entailment",
                    "metadata": {"data_classification_policy": ["public"]},
                    "label": "entailment",
                },
            ],
        }

        stream = recipe()
        result = stream["train"].peek()

        result_task_data = json.loads(result.pop("task_data"))

        self.assertDictEqual(result, target)
        self.assertDictEqual(target_task_data, result_task_data)

    def test_random_template(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template=[
                "templates.key_val",
                "templates.classification.multi_class.relation.truthfulness.flan_5",
            ],
            format="formats.user_agent",
            demos_pool_size=100,
            num_demos=3,
        )

        target = {
            "metrics": ["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
            "data_classification_policy": ["public"],
            "target": "not entailment",
            "references": ["not entailment"],
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
            "source": '<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n\n\n\nUser: Problem: Sentence: "Emma did not pass the ball to Janie although she was open.";\nAnother sentence: "She saw that Janie was open."?\nAgent: A: not entailment\n\nUser: Problem: Sentence: "The foxes are getting in at night and attacking the chickens. I shall have to kill them.";\nAnother sentence: "I shall have to kill The foxes."?\nAgent: A: not entailment\n\nUser: Problem: Sentence: "Fred is the only man alive who still remembers my father as an infant. When Fred first saw my father, he was twelve years old.";\nAnother sentence: "When Fred first saw my father, My father was twelve years old."?\nAgent: A: entailment\n\n\nUser:Problem: Sentence: "Grace was happy to trade me her sweater for my jacket. She thinks it looks dowdy on her.";\nAnother sentence: "The sweater looks dowdy on her."?\nAgent:A: ',
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": [],
        }

        stream = recipe()
        result = stream["train"].peek()

        result.pop("task_data")

        self.assertDictEqual(result, target)

    def test_random_template_with_templates_list(self):
        templates = TemplatesList(
            items=[
                "templates.key_val",
                "templates.classification.multi_class.relation.truthfulness.flan_5",
            ]
        )
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template=templates,
            format="formats.user_agent",
            demos_pool_size=100,
            num_demos=3,
        )

        target = {
            "metrics": ["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
            "data_classification_policy": ["public"],
            "target": "not entailment",
            "references": ["not entailment"],
            "postprocessors": [
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
            "source": '<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n\n\n\nUser: Problem: Sentence: "Emma did not pass the ball to Janie although she was open.";\nAnother sentence: "She saw that Janie was open."?\nAgent: A: not entailment\n\nUser: Problem: Sentence: "The foxes are getting in at night and attacking the chickens. I shall have to kill them.";\nAnother sentence: "I shall have to kill The foxes."?\nAgent: A: not entailment\n\nUser: Problem: Sentence: "Fred is the only man alive who still remembers my father as an infant. When Fred first saw my father, he was twelve years old.";\nAnother sentence: "When Fred first saw my father, My father was twelve years old."?\nAgent: A: entailment\n\n\nUser:Problem: Sentence: "Grace was happy to trade me her sweater for my jacket. She thinks it looks dowdy on her.";\nAnother sentence: "The sweater looks dowdy on her."?\nAgent:A: ',
            "groups": [],
            "media": {"audios": [], "images": []},
            "subset": [],
        }

        stream = recipe()
        result = stream["train"].peek()

        result.pop("task_data")

        self.assertDictEqual(result, target)

    def test_random_num_demos(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            demos_pool_size=100,
            num_demos=[0, 1, 3, 5],
        )

        stream = recipe()
        lengths = set()
        for i, instance in enumerate(stream["train"]):
            if i > 30:
                break
            lengths.add(len(instance["source"].split("\nAgent:")))

        self.assertEqual(len(lengths), 4)

    def test_standard_recipe_with_balancer(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            train_refiner="operators.balancers.classification.by_label",
            demos_pool_size=100,
            num_demos=3,
        )

        stream = recipe()
        counts = collections.Counter()
        for instance in stream["train"]:
            counts[instance["target"]] += 1

        self.assertEqual(counts["entailment"], counts["not entailment"])

    def test_standard_recipe_with_loader_limit(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            demos_pool_size=5,
            num_demos=1,
            loader_limit=10,
        )

        stream = recipe()
        self.assertEqual(
            len(list(stream["train"])), 5
        )  # 5 elements were moved to demo pool
        self.assertEqual(len(list(stream["test"])), 10)

    def test_standard_recipe_with_loader_limit_errors(self):
        with self.assertRaises(ValueError):
            StandardRecipeWithIndexes(
                card="cards.wnli",
                template="templates.key_val",
                max_test_instances=10,
                loader_limit=9,
            )

        with self.assertRaises(ValueError):
            StandardRecipeWithIndexes(
                card="cards.wnli",
                template="templates.key_val",
                max_train_instances=10,
                loader_limit=9,
            )
        with self.assertRaises(ValueError):
            StandardRecipeWithIndexes(
                template="templates.key_val",
                card="cards.wnli",
                max_validation_instances=10,
                loader_limit=9,
            )

        with self.assertRaises(ValueError):
            StandardRecipeWithIndexes(
                template="templates.key_val",
                card="cards.wnli",
                num_demos=3,
                demos_pool_size=10,
                loader_limit=9,
            )

    def test_standard_recipe_with_no_demos_to_take(self):
        recipe = StandardRecipeWithIndexes(
            template="templates.key_val",
            card="cards.xwinogrande.pt",
            num_demos=3,
            demos_pool_size=10,
        )
        with self.assertRaises(Exception) as cm:
            list(recipe()["test"])

        self.assertTrue(
            str(cm.exception).startswith(
                "Unable to fetch instances from 'demos_pool' to 'demos'"
            )
        )

        with self.assertRaises(Exception) as cm:
            recipe = StandardRecipeWithIndexes(
                template="templates.key_val",
                card="cards.xwinogrande.pt",
                num_demos=3,
                demos_pool_size=0,
            )

        self.assertEqual(
            str(cm.exception),
            "When using demonstrations both num_demos and demos_pool_size should be assigned with positive integers.",
        )

        with self.assertRaises(Exception) as cm:
            recipe = StandardRecipeWithIndexes(
                template="templates.key_val",
                card="cards.xwinogrande.pt",
                num_demos=30,
                demos_pool_size=10,
            )

        self.assertEqual(
            str(cm.exception),
            "num_demos (got: 30) should not exceed demos_pool_size (got: 10)",
        )

    def test_standard_recipe_with_no_test(self):
        recipe = StandardRecipeWithIndexes(
            template="templates.key_val",
            card="cards.xwinogrande.pt",
            num_demos=3,
            demos_pool_size=10,
            demos_taken_from="test",
        )
        results = list(recipe()["test"])
        self.assertTrue(len(results) > 0)

    def test_standard_recipe_with_template_errors(self):
        # Check either template or template index was specified , but not both
        with self.assertRaises(AssertionError) as cm:
            StandardRecipeWithIndexes(
                card="cards.wnli", template="templates.key_val", template_card_index=100
            )
        self.assertTrue(
            re.match(
                "Specify either template (.*) or template_card_index (.*) but not both",
                str(cm.exception),
            )
            is not None
        )

        # Also check if string index is used
        with self.assertRaises(AssertionError) as cm:
            StandardRecipeWithIndexes(
                card="cards.wnli",
                template="templates.key_val",
                template_card_index="illegal_template",
            )
        self.assertTrue(
            re.match(
                "Specify either template (.*) or template_card_index (.*) but not both",
                str(cm.exception),
            )
            is not None
        )

        # Return an error if index is not found in card
        with self.assertRaises(ValueError) as cm:
            StandardRecipeWithIndexes(
                card="cards.wnli", template_card_index="illegal_template"
            )
        self.assertTrue("not defined in card." in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            StandardRecipeWithIndexes(card="cards.wnli", template_card_index=100)
        self.assertTrue("not defined in card." in str(cm.exception))

    def test_standard_recipe_with_balancer_and_size_limit(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            train_refiner="operators.balancers.classification.by_label",
            demos_pool_size=100,
            max_train_instances=20,
            num_demos=3,
        )

        stream = recipe()
        counts = collections.Counter()
        for instance in stream["train"]:
            counts[instance["target"]] += 1

        self.assertEqual(counts["entailment"], counts["not entailment"], 10)

    def test_standard_recipe_with_augmentor_on_task_input(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.sst2",
            augmentor="augmentors.text.white_space",
            template_card_index=0,
            max_train_instances=0,
            max_test_instances=2,
        )
        stream = recipe()
        sample = list(stream["test"])[1]
        source = sample["source"]
        pattern = "Classify the sentiment of the following sentence to one of these options: ((negative, positive)|(positive, negative)). sentence: (.*)"
        result = re.match(pattern, sample["source"], re.DOTALL)
        assert result, f"Unable to find '{pattern}' in '{source}'"
        result = result.group(4)
        original_text = "unflinchingly bleak and desperate "
        assert (
            result != original_text
        ), f"Augmented text '{result}' is equal to text without '{original_text}' and was not augmented"
        normalized_output_source = result.split()
        normalized_input_source = original_text.split()
        assert (
            normalized_output_source == normalized_input_source
        ), f"{normalized_output_source} is not equal to f{normalized_input_source}"

    def test_standard_recipe_with_train_size_limit(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            demos_pool_size=3,
            max_train_instances=10,
            max_test_instances=5,
            num_demos=1,
        )

        stream = recipe()

        self.assertEqual(len(list(stream["train"])), 10)
        self.assertEqual(len(list(stream["test"])), 5)

    def test_recipe_with_hf_with_twice_the_same_instance_demos(self):
        from datasets import load_dataset

        d = load_dataset(
            dataset_file,
            "__type__=standard_recipe_with_indexes,card=cards.wnli,template=templates.classification.multi_class.relation.default,system_prompt=system_prompts.models.llama,demos_pool_size=5,num_demos=1",
            streaming=True,
            trust_remote_code=True,
        )

        iterator = iter(d["train"])
        next(iterator)
        print_dict(next(iterator))

    def test_recipe_loaded_from_arguments_and_overwrites_only(self):
        from unitxt import load_dataset

        dataset = load_dataset(
            "card=cards.copa,template=templates.qa.multiple_choice.with_context.no_intro.helm[enumerator=[option 1, option 2]],num_demos=1,demos_pool_size=10,format=formats.user_agent,max_train_instances=5"
        )

        iterator = iter(dataset["train"])
        first_inst = next(iterator)
        self.assertListEqual(["metrics.accuracy"], first_inst["metrics"])

    def test_standard_recipe_with_a_missing_sampler(self):
        """Check that initializing a recipe with a card that does not have a sampler raises an exception."""
        task_card, _ = copy.deepcopy(fetch_artifact("cards.sst2"))
        task_card.sampler = None
        with self.assertRaises(ValueError) as e:
            StandardRecipeWithIndexes(
                card=task_card,
                template_card_index=0,
                max_train_instances=0,
                max_test_instances=2,
                num_demos=1,
                demos_pool_size=10,
            )
        self.assertEqual(
            str(e.exception),
            "Unexpected None value for card.sampler. To use num_demos > 0, please set a sampler on the TaskCard.",
        )

    def test_set_serializer_from_recipe(self):
        instances = [
            {
                "table": {
                    "header": ["col1", "col2"],
                    "rows": [["val1", "val2"], ["val3"], ["val4"]],
                },
                "answer": "2",
            },
        ]

        class MyTableSerializer(SingleTypeSerializer):
            serialized_type = Table

            def serialize(self, value: Table, instance: Dict[str, Any]) -> str:
                return str(value)

        task = Task(
            input_fields={"table": Table},
            reference_fields={"answer": str},
            prediction_type=str,
            metrics=["metrics.accuracy"],
        )

        template = InputOutputTemplate(
            input_format="Solve: {table}\nAnswer: ",
            output_format="{answer}",
            postprocessors=[],
        )

        card = TaskCard(
            loader=LoadFromDictionary(data={"train": instances}),
            preprocess_steps=[],
            task=task,
        )

        recipe = StandardRecipe(
            card=card,
            template=template,
            serializer=TableSerializer(),
        )
        result = next(iter(recipe()["train"]))["source"]
        target = "Solve: col1,col2\nval1,val2\nval3\nval4\nAnswer: \n"
        self.assertEqual(result, target)

        recipe = StandardRecipe(
            card=card,
            template=template,
            serializer=MyTableSerializer(),
        )
        result = next(iter(recipe()["train"]))["source"]
        target = "Solve: {'header': ['col1', 'col2'], 'rows': [['val1', 'val2'], ['val3'], ['val4']]}\nAnswer: \n"
        self.assertEqual(result, target)
