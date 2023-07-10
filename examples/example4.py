from src.unitxt.common import (
    CommonRecipe,
)

from src.unitxt.catalog import add_to_catalog
from src.unitxt.load import load_dataset
from src.unitxt.text_utils import print_dict

recipe = CommonRecipe(
    card='watson_emotion_card',
    demos_pool_size=100,
    num_demos=3,
    template_item=0,
)

add_to_catalog(recipe, 'watson_emotion_3_shot', collection='recipes', overwrite=True)

dataset = load_dataset('watson_emotion_3_shot')

print_dict(dataset['train'][0])