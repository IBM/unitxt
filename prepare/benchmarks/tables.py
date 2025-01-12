import argparse
import json
import os.path

from tqdm import tqdm
from unitxt import add_to_catalog, evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.settings_utils import get_settings
from unitxt.standard import DatasetRecipe

# usage: python prepare/benchmarks/tables.py [-out_path OUT_PATH] [-models MODELS] [-cards CARDS]
# [-serializers SERIALIZERS] [-max_augmentors MAX_AUGMENTORS] [-max_pred_tokens MAX_PRED_TOKENS]
# [-num_demos NUM_DEMOS] [-recipes RECIPES_ONLY] [-debug DEBUG]

# constants and params
SERIALIZERS = {
    "html",
    "json",
    "markdown",
    "indexed_row_major",
    "df",
    "concat",
    "csv",
}
TABLE_AUGMENTORS = {
    "shuffle_cols",
    "shuffle_rows",
    "transpose",
    "insert_empty_rows[times=2]",
    # "duplicate_columns",  # TODO: we leave lossy perturb for now
    # "duplicate_rows",
    # "shuffle_cols_names",
    # "mask_cols_names",
}
DESCRIPTIVE_DATASETS = {
    "scigen",
    "numeric_nlg",
    "qtsumm",
    "tablebench_visualization",
    "tablebench_data_analysis",
}  # for making max pred tokens bigger
DATASET_WITH_LONG_EXAMPLES = {
    # "wikitq"
}

# TODO: Can we consider these parameters as final? the test sets are build from val+test as a part of the cards
DEMOS_POOL_SIZE = -1
MAX_PREDICTIONS = 100
LOADER_LIMIT = 10000
TEMPERATURE = 0.05
MAX_REQUESTS_PER_SECOND = 6
COMB_SIZE_AUGMENT = 1

settings = get_settings()

# cmd params
parser = argparse.ArgumentParser()
parser.add_argument(
    "-out_path", "--out_path", type=str, required=False, default="debug"
)
parser.add_argument(
    "-models",
    "--models",
    type=str,
    required=False,
    default="meta-llama/llama-3-1-70b-instruct",
)
parser.add_argument(
    "-cards",
    "--cards",
    type=str,
    required=False,
    default="fin_qa,wikitq,turl_col_type,tab_fact,numeric_nlg,qtsumm,tablebench_data_analysis,scigen,"
    "tablebench_fact_checking,tablebench_numerical_reasoning",  # ,tablebench_visualization",
)  # TODO: Scigen is implemented with 1 judge. Should we make it 3?
parser.add_argument(
    "-serializers",
    "--serializers",
    type=str,
    required=False,
    default=",".join(list(SERIALIZERS)),
)
parser.add_argument(
    "-max_augmentors",
    "--max_augmentors",
    type=int,
    required=False,
    default=10,
)
parser.add_argument(
    "-max_pred_tokens", "--max_pred_tokens", type=int, required=False, default=100
)  # We set it to be 300 for descriptive tasks (numeric nlg, scigen, qtsumm)
parser.add_argument(
    "-num_demos", "--num_demos", type=int, required=False, default=5
)  # num of demos for wikitq is 1 (tables with many rows)
parser.add_argument("-recipes_only", "--recipes_only", type=bool, default=False)
parser.add_argument("-debug", "--debug", type=bool, default=False)

args = parser.parse_args()
models = args.models
num_demos = args.num_demos
out_path = args.out_path
debug = args.debug
cards = args.cards
serializers = args.serializers
max_augmentors = args.max_augmentors
max_pred_tokens = args.max_pred_tokens
recipes_only = args.recipes_only

# formatting cmd params
models_parsed = [item.strip() for item in models.split(",")]
cards_parsed = [item.strip() for item in cards.split(",")]
serializers_parsed = [item.strip() for item in serializers.split(",")]
all_augment = [None] + [[i] for i in TABLE_AUGMENTORS]


# create the recipes subset dynamically
def get_recipes():
    recipes = {}
    for card in cards_parsed:
        for augment in all_augment:
            for serializer in serializers_parsed:
                subset_name = (
                    "dataset="
                    + card
                    + (
                        "__serializer=" + serializer
                        if serializer in SERIALIZERS
                        else ""
                    )
                    + ("__augment=" + ",".join(augment) if augment else "")
                    + "__num_demos="
                    + str(num_demos)
                )

                kwargs = {
                    "card": "cards." + card,
                    "serializer": "serializers.table." + serializer
                    if serializer in SERIALIZERS and serializer != "csv"
                    else None,
                    "num_demos": num_demos,
                    "demos_pool_size": DEMOS_POOL_SIZE,
                    "loader_limit": LOADER_LIMIT,
                    "augmentor": [
                        "augmentors."
                        + ("table." if a in TABLE_AUGMENTORS else "")
                        + str(a)
                        for a in augment
                    ]
                    if augment
                    else None,
                }

                def stringify_kwarg_value(value) -> str:
                    if isinstance(value, list):
                        return f"[{','.join(value)}]"
                    return value

                str_recipe = ",".join(
                    f"{key}={stringify_kwarg_value(value)}"
                    for key, value in kwargs.items()
                )

                obj_recipe = DatasetRecipe(**kwargs)

                recipes[subset_name] = str_recipe if recipes_only else obj_recipe

                add_to_catalog(
                    obj_recipe,
                    "recipes.tables_benchmark."
                    + card
                    + "."
                    + serializer
                    + "."
                    + (",".join(augment).split("[")[0] if augment else "no")
                    + "_augmentation_"
                    + str(num_demos)
                    + "_demos",
                    overwrite=True,
                )

    return recipes


if recipes_only:
    get_recipes()

elif len(models_parsed) > 0:  # run  benchmark
    # print("Run Params:", [f"{arg}: {value}" for arg, value in vars(args).items()])
    with settings.context(
        disable_hf_datasets_cache=False,
    ):
        # creating the subsets dynamically
        subsets = get_recipes()

        for model in models_parsed:
            model_name = model.split("/")[-1]

            # running one subset at a time and saving it separately
            for subset_name, subset in tqdm(subsets.items(), desc="Running..."):
                tqdm.write(f"Running subset: {subset_name}, Model: {model}")

                out_file_name = (
                    "model="
                    + model_name
                    + "__"
                    + subset_name
                    + ("__DEBUG" if debug else "")
                    + ".json"
                )
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                elif out_file_name.split("__num_demos")[0] in {
                    o.split("__num_demos")[0] for o in os.listdir(out_path)
                }:  # avoid running the same config twice
                    continue
                max_pred_tokens = (
                    300
                    if any(s in subset_name for s in DESCRIPTIVE_DATASETS)
                    else max_pred_tokens
                )

                benchmark = Benchmark(
                    max_samples_per_subset=MAX_PREDICTIONS if not debug else 5,
                    loader_limit=LOADER_LIMIT if not debug else 500,
                    subsets={subset_name: subset},
                )

                test_dataset = benchmark()["test"].to_dataset()

                try:
                    # inference_model = LiteLLMInferenceEngine(
                    #     model=model,
                    #     max_tokens=max_pred_tokens,
                    #     max_requests_per_second=MAX_REQUESTS_PER_SECOND,
                    #     format=formats.chat_api,
                    #     temperature=TEMPERATURE,
                    #
                    # )

                    # inference_model = IbmGenAiInferenceEngine(
                    #     model_name=model,
                    #     max_new_tokens=max_pred_tokens,
                    #     temperature=TEMPERATURE,
                    # )

                    # inference_model = RITSInferenceEngine(
                    #     model_name=model,
                    #     max_tokens=max_pred_tokens,
                    #     temperature=TEMPERATURE,
                    # )

                    inference_model = CrossProviderInferenceEngine(
                        model=model,
                        max_tokens=max_pred_tokens,
                        provider="rits",
                    )

                    predictions = inference_model.infer(test_dataset)
                    evaluated_dataset = evaluate(
                        predictions=predictions, data=test_dataset
                    )
                    for i in evaluated_dataset:
                        i.pop("postprocessors")

                    curr_out_path = os.path.join(out_path, out_file_name)
                    with open(curr_out_path, "w") as f:
                        json.dump(evaluated_dataset, f)
                        # print("saved file path: ", curr_out_path)
                except Exception as e:
                    with open(
                        os.path.join(out_path, "model=" + model_name + "__errors.txt"),
                        "a",
                    ) as f:
                        f.write("\n\n" + "\n".join([subset_name, str(e)]))
                    # print(e)
                    pass
