import collections
import unittest

from src.unitxt import dataset_file
from src.unitxt.formats import ICLFormat
from src.unitxt.instructions import TextualInstruction
from src.unitxt.standard import StandardRecipe, StandardRecipeWithIndexes
from src.unitxt.templates import InputOutputTemplate
from src.unitxt.text_utils import print_dict


class TestRecipes(unittest.TestCase):
    def test_standard_recipe(self):
        recipe = StandardRecipe(
            card="cards.wnli",
            instruction=TextualInstruction(text="classify"),
            template=InputOutputTemplate(
                input_format="{premise}",
                output_format="{label}",
            ),
            format=ICLFormat(
                input_prefix="User:",
                output_prefix="Agent:",
            ),
        )
        stream = recipe()

        for instance in stream["train"]:
            print_dict(instance)
            break

    def test_standard_recipe_with_catalog(self):
        recipe = StandardRecipe(
            card="cards.mmlu.marketing",
            instruction="instructions.models.llama",
            template="templates.mmlu.lm_eval_harness",
            format="formats.user_agent",
            demos_pool_size=100,
            num_demos=3,
        )

        stream = recipe()

        for instance in stream["train"]:
            print_dict(instance)
            break

    def test_standard_recipe_with_indexes_with_catalog(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            instruction="instructions.models.llama",
            template_card_index=0,
            format="formats.user_agent",
            demos_pool_size=100,
            num_demos=3,
        )

        stream = recipe()

        for instance in stream["train"]:
            print_dict(instance)
            break

    def test_empty_template(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            instruction="instructions.models.llama",
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
            instruction="instructions.models.llama",
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
            instruction="instructions.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            train_refiner="operators.balancers.balanced_targets",
            demos_pool_size=100,
            num_demos=3,
        )

        stream = recipe()
        counts = collections.Counter()
        for instance in stream["train"]:
            counts[instance["target"]] += 1

        self.assertEqual(counts["entailment"], counts["not entailment"])

    def test_standard_recipe_with_balancer_and_size_limit(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            instruction="instructions.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            train_refiner="operators.balancers.balanced_targets",
            demos_pool_size=100,
            max_train_instances=20,
            num_demos=3,
        )

        stream = recipe()
        counts = collections.Counter()
        for instance in stream["train"]:
            counts[instance["target"]] += 1

        self.assertEqual(counts["entailment"], counts["not entailment"], 10)

    def test_standard_recipe_with_train_size_limit(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            instruction="instructions.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            demos_pool_size=100,
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
            "type=standard_recipe_with_indexes,card=cards.wnli,template_card_index=0,demos_pool_size=5,num_demos=5,instruction=instructions.models.llama",
            streaming=True,
        )

        iterator = iter(d["train"])
        next(iterator)
        print_dict(next(iterator))
