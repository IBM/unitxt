from src.unitxt.common import (
    CommonRecipe,
)

from src.unitxt.catalog import add_to_catalog
from src.unitxt.load import load_dataset
from src.unitxt.text_utils import print_dict
from src import unitxt

dataset = load_dataset('recipes.wnli_3_shot')

import evaluate

metric = evaluate.load(unitxt.metric_url)

results = metric.compute(predictions=['none' for t in dataset['test']], references=dataset['test'])

print_dict(results[0])