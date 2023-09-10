import unittest

from src.unitxt.formats import ICLFormat
from src.unitxt.instructions import TextualInstruction
from src.unitxt.standard import StandardRecipe
from src.unitxt.templates import InputOutputTemplate


class TestRecipes(unittest.TestCase):
    def test_standard_recipe(self):
        recipe = StandardRecipe(
            card="cards.wnli",
            instruction=TextualInstruction(text="classify"),
            template=InputOutputTemplate(
                input_format="{sentence1}",
                output_format="{label}",
            ),
            format=ICLFormat(
                input_prefix="User:",
                output_prefix="Agent:",
            ),
        )

        stream = recipe()

        for instance in stream["train"]:
            print(instance)
            break
