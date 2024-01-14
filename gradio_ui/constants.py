import os

CATALOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "unitxt",
    "catalog",
)

AUGMENTABLE_STR = "augmentable_inputs"
LOADER_LIMIT_STR = "loader_limit"
PROMPT_SOURCE_STR = "source"
PROMPT_METRICS_STR = "metrics"
PROPT_TARGET_STR = "target"
DEMOS_POOL_SIZE = 100
PROMPT_SAMPLE_SIZE = 10
