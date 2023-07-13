from src.unitxt.blocks import (ToString)

from src.unitxt.metric import MetricRecipe

from src.unitxt.metrics import (
    Accuracy,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.text_utils import print_dict

add_to_catalog(ToString(), 'processors.to_string', overwrite=True)
add_to_catalog(Accuracy(), 'metrics.accuracy', overwrite=True)

data = [
    {'group': 'group1', 'metrics': ['accuracy'], 'references':[3]},
    {'group': 'group1', 'metrics': ['accuracy'], 'references':[4]},
    {'group': 'group2', 'metrics': ['accuracy'], 'references':[3]},
    {'group': 'group2', 'metrics': ['accuracy'], 'references':[3]},
]

predictions = [4, 3, 3, 3]

recipe = MetricRecipe(predictions=predictions, original_data=data)

result = recipe(data)

for item in next(iter(result.values())):
    print_dict(item)
    print('\n')