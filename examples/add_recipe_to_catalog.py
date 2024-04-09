from unitxt import load
from unitxt.catalog import add_to_catalog
from unitxt.standard import StandardRecipe
from unitxt.text_utils import print_dict

recipe = StandardRecipe(
    card="cards.wnli",
    demos_pool_size=100,
    num_demos=3,
    template_card_index=0,
)

add_to_catalog(recipe, "recipes.wnli_3_shot", overwrite=True)

dataset = load("recipes.wnli_3_shot")

print_dict(dataset["train"][0])
