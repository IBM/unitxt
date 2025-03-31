import os
import subprocess
from datetime import date
from typing import Dict

from card_utils import add_card_to_catalog
from process_utils import (
    apply_question_crop_middle,
    apply_shuffle_sql,
    convert_to_jsonl_and_save_short_version,
    save_augmented_train_sets,
)
from unitxt import evaluate, load_dataset, settings
from unitxt.inference import CrossProviderInferenceEngine

SHORT_FILE_LENGTH = 100
USE_SHORT = True
NUM_DEMOS = 3


target_dir = "~/.cache/huggingface/datasets/unitxt"
if os.path.exists(target_dir):
    subprocess.run(
        ["rm -r", target_dir],
        check=True,
        capture_output=True,
        text=True,
    )

EXP_NAME = f"{date.today().strftime('%y%m%d')}_sanity"

MODEL = "llama-3-3-70b-instruct"

base_files: Dict = {
    "train": "ics/data/text2sql/bird/train.json",
    "test": "ics/data/text2sql/bird/test.json",
}

convert_to_jsonl_and_save_short_version(SHORT_FILE_LENGTH, base_files)

new_base_files = {}
if USE_SHORT:
    for k, v in base_files.items():
        new_base_files[f"{k}{'_short' if USE_SHORT else ''}"] = v.replace(
            ".json", "_short.jsonl"
        )

# Mapping manipulation names to functions
aug_funs = {
    "shuffle_sql": apply_shuffle_sql,
    "question_crop_middle_08": lambda data: apply_question_crop_middle(
        data, crop_ratio=0.8
    ),
    "question_crop_middle_05": lambda data: apply_question_crop_middle(
        data, crop_ratio=0.5
    ),
    "question_crop_middle_03": lambda data: apply_question_crop_middle(
        data, crop_ratio=0.5
    ),
}

train_split_name = f"train{'_short' if USE_SHORT else ''}"
test_split_name = f"test{'_short' if USE_SHORT else ''}"

for aug_name, aug_fun in aug_funs.items():
    new_base_files[f"{train_split_name}_{aug_name}"] = save_augmented_train_sets(
        {aug_name: aug_fun}, new_base_files[train_split_name]
    )[0]

train_splits = [
    split_name for split_name in new_base_files.keys() if "train" in split_name
]

card_names = []

test_datasets = {}


for train_split in train_splits:
    card_name = f"cards.text2sql.bird.ics.{EXP_NAME}.{train_split}"
    card_names.append(card_name)

    cur_files = {}  # Initialize an empty dictionary to store the results

    # Iterate through the key-value pairs of the original dictionary
    for k, v in new_base_files.items():
        if train_split == k:
            cur_files["train"] = v
        elif test_split_name == k:
            cur_files["test"] = v

    add_card_to_catalog(card_name, cur_files)

# Infer
inference_model = CrossProviderInferenceEngine(
    model=MODEL,
    max_tokens=256,
)

exp_artifacts = {}

for card_name in card_names:
    with settings.context(
        disable_hf_datasets_cache=True,
        allow_unverified_code=True,
    ):
        exp_artifacts[card_name] = {}
        exp_artifacts[card_name]["test_dataset"] = load_dataset(
            split="test",
            dataset_query=f"card={card_name}"
            ",template=templates.text2sql.you_are_given_with_hint_with_sql_prefix,"
            f"num_demos={NUM_DEMOS},demos_pool_size=100,demos_removed_from_data=True",
        )

    exp_artifacts[card_name]["predictions"] = inference_model.infer(
        exp_artifacts[card_name]["test_dataset"]
    )
    exp_artifacts[card_name]["evaluated_dataset"] = evaluate(
        predictions=exp_artifacts[card_name]["predictions"],
        data=exp_artifacts[card_name]["test_dataset"],
    )

# for card_name in card_names:
#     print(
#         f"{card_name}: {exp_artifacts[card_name]['evaluated_dataset'][0]['score']['global']['score']}"
#     )


# exp_artifacts["cards.text2sql.bird.ics.first_sanity.train_shuffle_sql"]["test_dataset"][
#     0
# ]["source"] == exp_artifacts["cards.text2sql.bird.ics.first_sanity.train"][
#     "test_dataset"
# ][0]["source"]

# with llama-3-3-70b-instruct
# num_of_instances (int):
#     100
# execution_accuracy (float):
#     0.44

# CI is 0.34,0.54
# like GPT4 that goes to 46 (rank 40 in the benchmark https://bird-bench.github.io/)
# makes sense

# from transformers import AutoModelForCausalLM, AutoTokenizer

# DEBUG_NUM_EXAMPLES = 2
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# test_dataset = test_dataset.select(range(DEBUG_NUM_EXAMPLES))
# predictions = tokenizer.batch_decode(
#     model.generate(
#         **tokenizer.batch_encode_plus(
#             test_dataset["source"], return_tensors="pt", padding=True
#         ),
#         max_length=2048,
#     ),
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=True,
# )
