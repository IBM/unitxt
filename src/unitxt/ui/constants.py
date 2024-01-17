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
EMPTY_SCORES_FRAME = list({"": ""}.items())
SCORE_FRAME_HEADERS = ["Score Name", "Score"]
UNITEXT_METRIC_STR = "unitxt/metric"
PREDICTIONS_IMPORTS_STR = """
import evaluate
from transformers import pipeline"""
DATASET_IMPORT_STR = "from datasets import load_dataset"
PREDICTION_CODE_STR = f"""
model = pipeline(model="google/{FLAN_T5_BASE}")
predictions = [output["generated_text"] for output in model(dataset["{PROMPT_SOURCE_STR}"],max_new_tokens={MAX_NEW_TOKENS})]
metric = evaluate.load("{UNITEXT_METRIC_STR}")
scores = metric.compute(predictions=predictions,references=dataset)

[print(item) for item in scores[0]['score']['global'].items()]
"""
