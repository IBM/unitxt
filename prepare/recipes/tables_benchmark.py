from unitxt import add_to_catalog
from unitxt.standard import DatasetRecipe

# Constants and defaults
SERIALIZERS = {"html", "json", "markdown", "indexed_row_major", "df", "concat", "csv"}
TABLE_AUGMENTORS = {
    "shuffle_cols",
    "shuffle_rows",
    "transpose",
    "insert_empty_rows[times=2]",
}
DEMOS_POOL_SIZE = -1
MAX_PREDICTIONS = 100
LOADER_LIMIT = 10000
COMB_SIZE_AUGMENT = 1

# Default parameters
cards = (
    "fin_qa,wikitq,turl_col_type,tab_fact,numeric_nlg,qtsumm,tablebench_data_analysis,scigen,"
    "tablebench_fact_checking,tablebench_numerical_reasoning"
)
serializers = ",".join(list(SERIALIZERS))
max_augmentors = 10
max_pred_tokens = 100
recipes_only = False

# Process parameters
cards_parsed = [item.strip() for item in cards.split(",")]
serializers_parsed = [item.strip() for item in serializers.split(",")]
all_augment = [None] + [[i] for i in TABLE_AUGMENTORS]

# Create the recipes subset dynamically

for card in cards_parsed:
    for augment in all_augment:
        for serializer in serializers_parsed:
            num_demos = 1 if card == "wikitq" else 5
            kwargs = {
                "card": "cards." + card,
                "serializer": f"serializers.table.{serializer}"
                if serializer in SERIALIZERS and serializer != "csv"
                else None,
                "num_demos": num_demos,
                "demos_pool_size": DEMOS_POOL_SIZE,
                "loader_limit": LOADER_LIMIT,
                "augmentor": [f"augmentors.table.{a!s}" for a in augment]
                if augment
                else None,
            }

            add_to_catalog(
                DatasetRecipe(**kwargs),
                f"recipes.tables_benchmark.{card}.{serializer}."
                + (",".join(augment).split("[")[0] if augment else "no")
                + f"_augmentation_{num_demos}_demos",
                overwrite=True,
            )
