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
PREDICTIONS_IMPORTS_STR = """
import evaluate
from unitxt import metric_url
from transformers import T5ForConditionalGeneration, T5Tokenizer"""
DATASET_IMPORT_STR = "from datasets import load_dataset"
PREDICTION_CODE_STR = f"""
flan_tokenizer = T5Tokenizer.from_pretrained(f"google/{FLAN_T5_BASE}")
flan_model = T5ForConditionalGeneration.from_pretrained(f"google/{FLAN_T5_BASE}")

prompt_texts = [prompt['{PROMPT_SOURCE_STR}'] for prompt in dataset['train']]
input_ids = flan_tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).input_ids
output = flan_model.generate(input_ids, max_new_tokens={MAX_NEW_TOKENS})
predictions = [flan_tokenizer.decode(output_item, skip_special_tokens=True) for output_item in output]

metric = evaluate.load(metric_url)
scores = metric.compute(predictions=predictions,references=dataset['train'])

[print(item) for item in scores[0]['score']['global'].items()]
"""
