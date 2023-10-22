import collections
import re
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
            instruction="instructions.models.llama",
            template="templates.key_val",
            format="formats.user_agent",
            demos_pool_size=5,
            num_demos=1,
            loader_limit=10,
        )

        stream = recipe()
        self.assertEqual(len(list(stream["train"])), 5)
        self.assertEqual(len(list(stream["test"])), 10)

    def test_standard_recipe_with_loader_limit_errors(self):
        with self.assertRaises(ValueError):
            recipe = StandardRecipeWithIndexes(
                card="cards.wnli",
                template="templates.key_val",
                max_test_instances=10,
                loader_limit=9,
            )

        with self.assertRaises(ValueError):
            recipe = StandardRecipeWithIndexes(
                card="cards.wnli",
                template="templates.key_val",
                max_train_instances=10,
                loader_limit=9,
            )
        with self.assertRaises(ValueError):
            recipe = StandardRecipeWithIndexes(
                template="templates.key_val",
                card="cards.wnli",
                max_validation_instances=10,
                loader_limit=9,
            )

        with self.assertRaises(ValueError):
            recipe = StandardRecipeWithIndexes(
                template="templates.key_val",
                card="cards.wnli",
                num_demos=3,
                demos_pool_size=10,
                loader_limit=9,
            )

    def test_standard_recipe_with_balancer_and_size_limit(self):
        recipe = StandardRecipeWithIndexes(
            card="cards.wnli",
            instruction="instructions.models.llama",
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
        pattern = "Given this sentence: (.*), classify if it is negative, positive.\s*"
        result = re.match(pattern, sample["source"], re.DOTALL)
        assert result, f"Unable to find '{pattern}' in '{source}'"
        result = result.group(1)
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
        original_source = list(recipe()["test"])[0]["source"]

        recipe = StandardRecipeWithIndexes(
            card="cards.sst2",
            augmentor="augmentors.augment_whitespace_model_input",
            template_card_index=0,
            max_train_instances=0,
            max_test_instances=1,
        )
        augmented_source = list(recipe()["test"])[0]["source"]

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
