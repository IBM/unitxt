from src.unitxt.catalog import add_to_catalog
from src.unitxt.load import load_dataset
from src.unitxt.standard import StandardRecipe
from src.unitxt.text_utils import print_dict

recipe = StandardRecipe(
    card="cards.wnli",
    demos_pool_size=100,
    num_demos=3,
    template_card_index=0,
)

add_to_catalog(recipe, "recipes.wnli_3_shot", overwrite=True)

dataset = load_dataset("recipes.wnli_3_shot")

print_dict(dataset["train"][0])
