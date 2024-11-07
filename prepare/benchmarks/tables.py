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
    SerializeTableAsDFLoader,
    SerializeTableAsHTML,
    SerializeTableAsIndexedRowMajor,
    SerializeTableAsJson,
    SerializeTableAsMarkdown,
    SerializeTableRowAsText,
)

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
    "-seeds", "--seeds", type=str, required=False, default=str(settings.seed)
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
parser.add_argument("-comb_size", "--comb_size", type=int, required=False, default=1)
args = parser.parse_args()
model_name = args.model
num_demos = args.num_demos
out_path = os.path.join(args.out_path, args.model.split("/")[-1])
debug = args.debug
seeds = args.seeds
cards = args.cards
serializers = args.serializers
augmentors = args.augmentors
comb_size = args.comb_size

serializers_map = {
    "html": SerializeTableAsHTML,
    "json": SerializeTableAsJson,
    "markdown": SerializeTableAsMarkdown,
    "row_indexed_major": SerializeTableAsIndexedRowMajor,
    "df": SerializeTableAsDFLoader,
    "text": SerializeTableRowAsText,
    "csv": None,
}
ALLOWED_AUGMENTORS = {
    "shuffle_cols_names",
    "mask_cols_names",
    "truncate_rows",
    "shuffle_cols",
    "shuffle_rows",
    "insert_empty_rows",
    "duplicate_columns",
    "duplicate_rows",
    "transpose",
}
DEMOS_POOL_SIZE = 10

cards_parsed = [item.strip() for item in cards.split(",")]
serializers_parsed = [item.strip() for item in serializers.split(",")]

augmentors_parsed = (
    {item.strip() for item in augmentors.split(",")} if augmentors else {}
)
if augmentors_parsed and any(augmentors_parsed - ALLOWED_AUGMENTORS):
    # print(augmentors_parsed - ALLOWED_AUGMENTORS, "NOT ALLOWED !")
    augmentors_parsed = augmentors_parsed.intersection(ALLOWED_AUGMENTORS)
    augmentors_full_names = ["augmentors.table." + a for a in augmentors_parsed]
    comb_augmentors = combinations(augmentors_full_names, comb_size)
else:
    comb_augmentors = [None]

try:
    seeds_parsed = [int(i.strip()) for i in seeds.split(",")]
except:
    seeds_parsed = [settings.seed]

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
        + (
            "__augment="
            + "+".join([] if not augment else [aug.split(".")[-1] for aug in augment])
            if augment
            else ""
        )
    )


subsets = {}
for card in cards_parsed:
    for augment in comb_augmentors:
        for serializer in serializers_parsed:
            # for seed in seeds_parsed:
            subsets[get_subset_name(card, serializer, augment)] = StandardRecipe(
                card="cards." + card,
                template_card_index=0,
                serializer=serializers_map[serializer]()
                if serializers_map[serializer]
                else None,
                num_demos=num_demos,
                demos_pool_size=DEMOS_POOL_SIZE,
                format=format,
                augmentor=augment,
            )


# print("Run Params:", [f"{arg}: {value} | " for arg, value in vars(args).items()])
for subset_name, subset in tqdm(subsets.items()):
    # print(
    #     "Running:",
    #     subset_name,
    # )

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
                max_tokens=100,
                temperature=0.05,
            )
        else:
            inference_model = IbmGenAiInferenceEngine(
                model_name=model_name,
                max_new_tokens=100,
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
    except Exception:
        # print(e)
        pass
