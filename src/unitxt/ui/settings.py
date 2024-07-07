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
PROMPT_TARGET_STR = "target"
DEMOS_POOL_SIZE = 10
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
HEADER_VISIBLE = True
INTRO_TXT = """
# Data Preparation and Evaluation for Generative AI

## Flexible, Shareable and Reusable

[![unitxt](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/IBM/unitxt)
"""
JSON_INTRO_TXT = """
# Click the view button of a catalog item to see it
"""
CODE_INTRO_TXT = """
# When you generate prompts / infer with model,
# code to reproduce will show here
"""

MAIN_INTRO_TXT = """
# Select _Task_, _Dataset_ and a _Template_,
# Then click the __Generate__ __Prompts__ button!
"""
JSON_BUTTON_TXT = "View"

ACK_TEXT = """
Unitxt emoji designed by [OpenMoji](https://openmoji.org/#) - the open-source emoji and icon project.
 License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/#)
"""
