import argparse
import json
import os.path
import random
from itertools import combinations

from tqdm import tqdm
from unitxt import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    IbmGenAiInferenceEngine,
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

# usage: python prepare/benchmarks/tables.py -models MODELS -out_path OUT_PATH [-cards CARDS] [-serializers SERIALIZERS]
# [-num_augmentors NUM_AUGMENTORS] [-max_pred_tokens MAX_PRED_TOKENS] [-num_demos NUM_DEMOS] [-debug DEBUG]

# constants and params
SERIALIZERS_MAP = {
    "html": SerializeTableAsHTML,
    "json": SerializeTableAsJson,
    "markdown": SerializeTableAsMarkdown,
    "row_indexed_major": SerializeTableAsIndexedRowMajor,
    "df": SerializeTableAsDFLoader,
    "concat": SerializeTableAsConcatenation,
    "csv": None,
}
TABLE_AUGMENTORS = [
    "shuffle_cols",
    "shuffle_rows",
    "transpose",
    "insert_empty_rows",
    # "duplicate_columns",  # TODO: we leave lossy perturb for now
    # "duplicate_rows",
    # "shuffle_cols_names",
    # "mask_cols_names",
]
DESCRIPTIVE_DATASETS = {
    "scigen",
    "numeric_nlg",
    "qtsumm",
}  # for making max pred tokens bigger

# TODO: Can we consider these parameters as final? the test sets are build from val+test as a part of the cards
DEMOS_POOL_SIZE = 10
MAX_PREDICTIONS = 100
LOADER_LIMIT = 500
TEMPERATURE = 0.05
MAX_REQUESTS_PER_SECOND = 6
COMB_SIZE_AUGMENT = 1

settings = get_settings()

# cmd params
parser = argparse.ArgumentParser()
parser.add_argument("-models", "--models", type=str, required=True)
parser.add_argument("-out_path", "--out_path", type=str, required=True)
parser.add_argument(
    "-cards",
    "--cards",
    type=str,
    required=False,
    default="fin_qa,wikitq,turl_col_type,tab_fact,numeric_nlg,qtsumm,tablebench_data_analysis,scigen,"
    "tablebench_fact_checking,tablebench_numerical_reasoning,tablebench_visualization",
)  # TODO: Scigen is implemented with 1 judge. Should we make it 3?
parser.add_argument(
    "-serializers",
    "--serializers",
    type=str,
    required=False,
    default="html,csv,json,markdown,row_indexed_major,df,concat",
)
parser.add_argument(
    "-num_augmentors",
    "--num_augmentors",
    type=int,
    required=False,
    default=10,
)
parser.add_argument(
    "-max_pred_tokens", "--max_pred_tokens", type=int, required=False, default=100
)  # TODO: Should we set a different num for descriptive tasks? (numeric nlg, scigen, qtsumm)
parser.add_argument("-num_demos", "--num_demos", type=int, required=False, default=5)
parser.add_argument("-debug", "--debug", type=bool, default=False)

args = parser.parse_args()
models = args.models
num_demos = args.num_demos
out_path = args.out_path
debug = args.debug
cards = args.cards
serializers = args.serializers
num_augmentors = args.num_augmentors
max_pred_tokens = args.max_pred_tokens

# formatting cmd params
models_parsed = [item.strip() for item in models.split(",")]
cards_parsed = [item.strip() for item in cards.split(",")]
serializers_parsed = [item.strip() for item in serializers.split(",")]

augment_combinations = list(combinations(TABLE_AUGMENTORS, COMB_SIZE_AUGMENT))
random.seed(settings.seed)
rand_augment_combinations = random.sample(
    augment_combinations, min(num_augmentors, len(augment_combinations))
)
all_augment = (
    [[None]] + [list(i) for i in rand_augment_combinations]
)  # TODO: Do we want other augmentations/sampling? Also we cannot augment with params for now

# TODO: Yifan - you can decide if to send runs in parallel or not (the loops exist for each case)
# running benchmark
# print("Run Params:", [f"{arg}: {value}" for arg, value in vars(args).items()])
with settings.context(
    disable_hf_datasets_cache=False,
):
    for model in models_parsed:
        model_name = model.split("/")[-1]

        format = "formats.empty"
        if "llama" in model:
            format = (
                "formats.llama3_instruct_all_demos_in_one_turn_without_system_prompt"
            )
        elif "mixtral" in model:
            format = "formats.models.mistral.instruction.all_demos_in_one_turn"

        # creating the subsets dynamically
        subsets = {}
        for card in cards_parsed:
            for augment in all_augment:
                for serializer in serializers_parsed:
                    subset_name = (
                        "dataset="
                        + card
                        + "__serializer="
                        + serializer
                        + (
                            "__augment=" + ",".join(augment)
                            if augment != [None]
                            else ""
                        )
                    )
                    subsets[subset_name] = StandardRecipe(
                        card="cards." + card,
                        template_card_index=0,
                        serializer=SERIALIZERS_MAP[serializer]()
                        if serializer in SERIALIZERS_MAP and SERIALIZERS_MAP[serializer]
                        else None,
                        num_demos=num_demos,
                        demos_pool_size=DEMOS_POOL_SIZE,
                        format=format,
                        augmentor=[None]
                        if augment == [None]
                        else [
                            "augmentors."
                            + ("table." if a in TABLE_AUGMENTORS else "")
                            + str(a)
                            for a in augment
                        ],
                    )

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
            elif out_file_name in os.listdir(
                out_path
            ):  # avoid running the same config twice
                continue
            max_pred_tokens = (
                300
                if any(s in subset_name for s in DESCRIPTIVE_DATASETS)
                else max_pred_tokens
            )

            benchmark = Benchmark(
                max_samples_per_subset=MAX_PREDICTIONS if not debug else 5,
                loader_limit=LOADER_LIMIT if not debug else 100,
                subsets={subset_name: subset},
            )

            test_dataset = benchmark()["test"].to_dataset()

            try:
                # TODO: Yifan please replace LiteLLMInferenceEngine with IbmGenAiInferenceEngine and test it.
                # inference_model = LiteLLMInferenceEngine(
                #     model=model,
                #     max_tokens=max_pred_tokens,
                #     max_requests_per_second=MAX_REQUESTS_PER_SECOND,
                #     format=formats.chat_api,
                #     temperature=TEMPERATURE,
                #
                # )

                inference_model = IbmGenAiInferenceEngine(
                    model_name=model,
                    max_new_tokens=max_pred_tokens,
                    temperature=TEMPERATURE,
                )

                predictions = inference_model.infer(test_dataset)
                evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)
                for i in evaluated_dataset:
                    i.pop("postprocessors")

                curr_out_path = os.path.join(out_path, out_file_name)
                with open(curr_out_path, "w") as f:
                    json.dump(evaluated_dataset, f)
                    # print("saved file path: ", curr_out_path)
            except Exception as e:
                with open(
                    os.path.join(out_path, "model=" + model_name + "__errors.txt"), "a"
                ) as f:
                    f.write("\n".join([subset_name, str(e)]))
                # print(e)
                pass
