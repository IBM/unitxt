import os

CATALOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "catalog",
)
BANNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banner.png")
AUGMENTABLE_STR = "augmentable_inputs"
LOADER_LIMIT_STR = "loader_limit"
PROMPT_SOURCE_STR = "source"
PROMPT_METRICS_STR = "metrics"
PROPT_TARGET_STR = "target"
DEMOS_POOL_SIZE = 100
PROMPT_SAMPLE_SIZE = 5
MAX_NEW_TOKENS = 30
FLAN_T5_BASE = "flan-t5-base"
GPT2 = "gpt2"
