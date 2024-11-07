import argparse
import os.path
import pickle
from itertools import combinations

import torch
from tqdm import tqdm
from unitxt import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    OpenAiInferenceEngine,
)
from unitxt.settings_utils import get_settings
from unitxt.standard import StandardRecipe
from unitxt.struct_data_operators import (
    SerializeTableAsConcatenation,
    SerializeTableAsDFLoader,
    SerializeTableAsHTML,
    SerializeTableAsIndexedRowMajor,
    SerializeTableAsJson,
    SerializeTableAsMarkdown,
)

# usage: tables.py [-h] -model MODEL -num_demos NUM_DEMOS -out_path OUT_PATH [-debug DEBUG] -cards CARDS [
# -serializers SERIALIZERS] [-augmentors AUGMENTORS] [-augmentors_comb_size AUGMENTORS_COMB_SIZE] [-max_pred_tokens
# MAX_PRED_TOKENS]

settings = get_settings()

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", type=str, required=True)
parser.add_argument("-num_demos", "--num_demos", type=int, required=True)
parser.add_argument("-out_path", "--out_path", type=str, required=True)
parser.add_argument("-debug", "--debug", type=bool, default=False)
parser.add_argument(
    "-cards",
    "--cards",
    type=str,
    required=True,
)
parser.add_argument(
    "-serializers",
    "--serializers",
    type=str,
    required=False,
    default="html,csv,json,markdown,row_indexed_major,df,concat",
)
parser.add_argument(
    "-augmentors",
    "--augmentors",
    type=str,
    required=False,
    default=None,
)
parser.add_argument(
    "-augmentors_comb_size",
    "--augmentors_comb_size",
    type=int,
    required=False,
    default=1,
)
parser.add_argument(
    "-max_pred_tokens", "--max_pred_tokens", type=int, required=False, default=100
)
args = parser.parse_args()
model_name = args.model
num_demos = args.num_demos
out_path = os.path.join(args.out_path, args.model.split("/")[-1])
debug = args.debug
cards = args.cards
serializers = args.serializers
augmentors = args.augmentors
augmentors_comb_size = args.augmentors_comb_size
max_pred_tokens = args.max_pred_tokens

SERIALIZERS_MAP = {
    "html": SerializeTableAsHTML,
    "json": SerializeTableAsJson,
    "markdown": SerializeTableAsMarkdown,
    "row_indexed_major": SerializeTableAsIndexedRowMajor,
    "df": SerializeTableAsDFLoader,
    "concat": SerializeTableAsConcatenation,
    "csv": None,
}
ALLOWED_AUGMENTORS = {
    "shuffle_cols_names",
    "mask_cols_names",
    "shuffle_cols",
    "shuffle_rows",
    "insert_empty_rows",
    "duplicate_columns",
    "duplicate_rows",
    "transpose",
}
DEMOS_POOL_SIZE = 10

# def select_random_combinations(lst, num_combinations=20):
#     # Generate all non-empty subsets
#     non_empty_subsets = []
#     for r in range(1, len(lst) + 1):
#         non_empty_subsets.extend(itertools.combinations(lst, r))
#
#     # Randomly select num_combinations subsets
#     selected_combinations = random.sample(non_empty_subsets, num_combinations)
#
#     # Convert the combinations from tuples back to lists (optional, but can be useful)
#     selected_combinations = [list(comb) for comb in selected_combinations]
#
#     # Sort the combinations by their length
#     selected_combinations.sort(key=len)
#
#     return selected_combinations

cards_parsed = [item.strip() for item in cards.split(",")]
serializers_parsed = [item.strip() for item in serializers.split(",")]
augmentors_parsed = (
    {item.strip() for item in augmentors.split(",")} if augmentors else {}
)
if augmentors_parsed:
    # if any(augmentors_parsed - ALLOWED_AUGMENTORS):
    #     print(augmentors_parsed - ALLOWED_AUGMENTORS, "NOT ALLOWED !")
    augmentors_parsed = augmentors_parsed.intersection(ALLOWED_AUGMENTORS)
    comb_augmentors = [
        list(comb) for comb in combinations(augmentors_parsed, augmentors_comb_size)
    ]
else:
    comb_augmentors = [["shuffle_cols_names"], ["duplicate_rows", "transpose"]]

format = "formats.empty"
if "llama" in model_name:
    format = "formats.llama3_instruct_all_demos_in_one_turn_without_system_prompt"
elif "mixtral" in model_name:
    format = "formats.models.mistral.instruction.all_demos_in_one_turn"


def get_subset_name(card, serializer, augment, seed=settings.seed):
    return (
        "dataset="
        + card
        + "__serializer="
        + serializer
        + ("__seed=" + str(seed) if seed != settings.seed else "")
        + ("__augment=" + ",".join(augment) if augment else "")
    )


subsets = {}
for card in cards_parsed:
    for augment in comb_augmentors:
        for serializer in serializers_parsed:
            subsets[get_subset_name(card, serializer, augment)] = StandardRecipe(
                card="cards." + card,
                template_card_index=0,
                serializer=SERIALIZERS_MAP[serializer]()
                if serializer in SERIALIZERS_MAP and SERIALIZERS_MAP[serializer]
                else None,
                num_demos=num_demos,
                demos_pool_size=DEMOS_POOL_SIZE,
                format=format,
                augmentor=["augmentors.table." + a for a in augment]
                if augment
                else [None],
            )


# print("Run Params:", [f"{arg}: {value} | " for arg, value in vars(args).items()])
for subset_name, subset in tqdm(subsets.items()):
    # print("Running:", subset_name)

    benchmark = Benchmark(
        max_samples_per_subset=100 if not debug else 5,
        loader_limit=500 if not debug else 100,
        subsets={subset_name: subset},
    )

    test_dataset = list(benchmark()["test"])

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    try:
        torch.cuda.empty_cache()
        if "gpt" in model_name:
            inference_model = OpenAiInferenceEngine(
                model_name=model_name,
                max_tokens=max_pred_tokens,
                temperature=0.05,
            )
        else:
            inference_model = IbmGenAiInferenceEngine(
                model_name=model_name,
                max_new_tokens=max_pred_tokens,
                temperature=0.05,
            )

        predictions = inference_model.infer(test_dataset)
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

        out_file_name = (
            model_name.split("/")[-1] + "#" + subset_name + ("__DEBUG" if debug else "")
        )
        curr_out_path = os.path.join(out_path, out_file_name) + ".pkl"
        with open(curr_out_path, "wb") as f:
            pickle.dump(evaluated_dataset, f)
            # print("saved file path: ", curr_out_path)
    except Exception as e:
        with open(os.path.join(out_path, "errors.txt"), "a") as f:
            f.write(str(e))
        # print(e)
        pass
