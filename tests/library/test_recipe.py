import collections
import copy
import re

from unitxt import dataset_file
from unitxt.artifact import fetch_artifact
from unitxt.formats import SystemFormat
from unitxt.standard import StandardRecipe, StandardRecipeWithIndexes
from unitxt.templates import InputOutputTemplate
from unitxt.text_utils import print_dict

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
                    "group": "unitxt",
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
                    "answer": "maybe",
                    "topic": "testing",
                }
            ]
        )[0]

        target = {
            "metrics": ["metrics.accuracy"],
            "source": "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n\n\n\n\nUser:The following are multiple choice questions (with answers) about testing.\n\nwhat?\nA. yes\nB. not\nC. maybe\nAnswer:\nAgent:",
            "target": " C",
            "references": [" C"],
            "task_data": '{"topic": "testing", '
            '"question": "what?", '
            '"choices": ["yes", "not", "maybe"], '
            '"answer": "maybe", '
            '"options": [" A", " B", " C"], '
            '"metadata": {"template": "templates.qa.multiple_choice.with_topic.lm_eval_harness"}'
            "}",
            "group": "unitxt",
            "postprocessors": ["processors.first_character"],
            "data_classification_policy": [],
        }

        self.assertDictEqual(result, target)

    def test_standard_recipe_production_consistency(self):
        recipe = StandardRecipe(
            card="cards.mmlu.marketing",
            system_prompt="system_prompts.models.llama",
            template="templates.qa.multiple_choice.with_topic.lm_eval_harness",
            format="formats.user_agent",
            demos_pool_size=5,
            num_demos=3,
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
                    "answer": "maybe",
                    "topic": "testing",
                }
            ]
        )[0]

        target = {
            "metrics": ["metrics.accuracy"],
            "source": "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n\n\n\nUser: The following are multiple choice questions (with answers) about marketing.\n\nThe single group within society that is most vulnerable to reference group influence is:\nA. The older consumer who feels somewhat left out of things.\nB. The married women, many of whom feel a need for stability in their lives.\nC. New immigrants who really want to assimilate into their new culture.\nD. Children, who base most of their buying decisions on outside influences.\nAnswer:\nAgent:  D\n\nUser: The following are multiple choice questions (with answers) about marketing.\n\n Which of the following is an assumption in Maslow's hierarchy of needs?\nA. Needs are dependent on culture and also on social class.\nB. Lower-level needs must be at least partially satisfied before higher needs can affect behaviour.\nC. Needs are not prioritized or arranged in any particular order.\nD. Satisfied needs are motivators, and new needs emerge when current needs remain unmet.\nAnswer:\nAgent:  B\n\nUser: The following are multiple choice questions (with answers) about marketing.\n\nIn an organization, the group of people tasked with buying decisions is referred to as the _______________.\nA. Outsourcing unit.\nB. Procurement centre.\nC. Chief executive unit.\nD. Decision-making unit.\nAnswer:\nAgent:  D\n\n\nUser:The following are multiple choice questions (with answers) about testing.\n\nwhat?\nA. yes\nB. not\nC. maybe\nAnswer:\nAgent:",
            "target": " C",
            "references": [" C"],
            "task_data": '{"topic": "testing",'
            ' "question": "what?",'
            ' "choices": ["yes", "not", "maybe"],'
            ' "answer": "maybe",'
            ' "options": [" A", " B", " C"],'
            ' "metadata": {"template": "templates.qa.multiple_choice.with_topic.lm_eval_harness"}'
            "}",
            "group": "unitxt",
            "postprocessors": ["processors.first_character"],
            "data_classification_policy": [],
        }

        self.assertDictEqual(result, target)

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

        stream = recipe()

        for instance in stream["train"]:
            print_dict(instance)
            break

    def test_key_val_template(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            demos_pool_size=100,
            num_demos=3,
        )

        stream = recipe()

        for instance in stream["train"]:
            print_dict(instance)
            break

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
        # Check some template was specified
        with self.assertRaises(AssertionError) as cm:
            StandardRecipeWithIndexes(card="cards.wnli")
        self.assertEqual(
            str(cm.exception), "Specify either template or template_card_index in card"
        )

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
            augmentor="augmentors.augment_whitespace_task_input",
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

    def test_standard_recipe_with_augmentor_on_model_input(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.sst2",
            template_card_index=0,
            max_train_instances=0,
            max_test_instances=1,
        )
        original_source = next(iter(recipe()["test"]))["source"]

        recipe = StandardRecipeWithIndexes(
            card="cards.sst2",
            augmentor="augmentors.augment_whitespace_model_input",
            template_card_index=0,
            max_train_instances=0,
            max_test_instances=1,
        )
        augmented_source = next(iter(recipe()["test"]))["source"]

        assert (
            original_source != augmented_source
        ), f"Augmented text '{augmented_source}' is equal to text without '{original_source}' and was not augmented"
        normalized_augmented_source = augmented_source.split()
        normalized_input_source = original_source.split()
        assert (
            normalized_augmented_source == normalized_input_source
        ), f"{normalized_augmented_source} is not equal to f{normalized_input_source}"

    def test_standard_recipe_with_train_size_limit(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            system_prompt="system_prompts.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            demos_pool_size=3,
            max_train_instances=10,
            max_test_instances=5,
            num_demos=3,
        )

        stream = recipe()

        self.assertEqual(len(list(stream["train"])), 10)
        self.assertEqual(len(list(stream["test"])), 5)

    def test_recipe_with_hf_with_twice_the_same_instance_demos(self):
        from datasets import load_dataset

        d = load_dataset(
            dataset_file,
            "type=standard_recipe_with_indexes,card=cards.wnli,template=templates.classification.multi_class.relation.default,system_prompt=system_prompts.models.llama,demos_pool_size=5,num_demos=5",
            streaming=True,
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

    def test_standard_recipe_with_a_sampler(self):
        """Check that the sampler is re-initialized before processing a recipe.

        To do so, save the random generator within the sampler before activating the recipe,
        and compare it to the random generator within the sampler after the revipe was called.
        The two generators should be different objects, indicating that the sampler was properly
        re-initialized during the preparation of the recipe.
        """
        recipe = StandardRecipeWithIndexes(
            card="cards.sst2",
            template_card_index=0,
            max_train_instances=0,
            max_test_instances=2,
            num_demos=1,
            demos_pool_size=10,
        )
        sampler = recipe.card.sampler

        random_generator1 = sampler.random_generator
        recipe()
        random_generator2 = sampler.random_generator

        self.assertNotEqual(random_generator1, random_generator2)

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
