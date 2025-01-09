import random
from itertools import combinations

from unitxt import add_to_catalog
from unitxt.standard import DatasetRecipe

# Constants and defaults
SERIALIZERS = {"html", "json", "markdown", "indexed_row_major", "df", "concat", "csv"}
TABLE_AUGMENTORS = {"transpose", "insert_empty_rows[times=2]"}
DESCRIPTIVE_DATASETS = {
    "scigen",
    "numeric_nlg",
    "qtsumm",
    "tablebench_visualization",
    "tablebench_data_analysis",
}
DEMOS_POOL_SIZE = 10
MAX_PREDICTIONS = 100
LOADER_LIMIT = 10000
COMB_SIZE_AUGMENT = 1

# Default parameters
out_path = "debug"
models = "meta-llama/llama-3-1-70b-instruct"
cards = (
    "fin_qa,wikitq,turl_col_type,tab_fact,numeric_nlg,qtsumm,tablebench_data_analysis,scigen,"
    "tablebench_fact_checking,tablebench_numerical_reasoning"
)
serializers = ",".join(list(SERIALIZERS))
max_augmentors = 10
max_pred_tokens = 100
num_demos = 5
recipes_only = False

# Process parameters
models_parsed = [item.strip() for item in models.split(",")]
cards_parsed = [item.strip() for item in cards.split(",")]
serializers_parsed = [item.strip() for item in serializers.split(",")]

augment_combinations = list(combinations(TABLE_AUGMENTORS, COMB_SIZE_AUGMENT))
random.seed(42)  # Replace with desired seed value
rand_augment_combinations = random.sample(
    augment_combinations, min(max_augmentors, len(augment_combinations))
)
all_augment = [None] + [list(i) for i in rand_augment_combinations]

# Create the recipes subset dynamically

for card in cards_parsed:
    demos_pool_size = DEMOS_POOL_SIZE
    for augment in all_augment:
        for serializer in serializers_parsed:
            curr_num_demos = num_demos

            kwargs = {
                "card": "cards." + card,
                "serializer": f"serializers.table.{serializer}"
                if serializer in SERIALIZERS and serializer != "csv"
                else None,
                "num_demos": curr_num_demos,
                "demos_pool_size": demos_pool_size,
                "augmentor": [f"augmentors.table.{a!s}" for a in augment]
                if augment
                else None,
            }

            add_to_catalog(
                DatasetRecipe(**kwargs),
                f"recipes.tables_benchmark.{card}.{serializer}."
                + (",".join(augment).split("[")[0] if augment else "no")
                + f"_augmentation_{curr_num_demos}_demos",
                overwrite=True,
            )
