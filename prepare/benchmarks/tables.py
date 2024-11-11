import argparse
import os.path
import pickle
import random
from itertools import chain, combinations

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

# usage: tables.py -models MODELS -out_path OUT_PATH [-num_demos NUM_DEMOS] [-cards CARDS] [-serializers SERIALIZERS]
# [-num_augmentors NUM_AUGMENTORS] [-max_pred_tokens MAX_PRED_TOKENS] [-debug DEBUG]


settings = get_settings()

parser = argparse.ArgumentParser()
parser.add_argument("-models", "--models", type=str, required=True)
parser.add_argument("-out_path", "--out_path", type=str, required=True)
parser.add_argument("-num_demos", "--num_demos", type=int, required=False, default=5)
parser.add_argument(
    "-cards",
    "--cards",
    type=str,
    required=False,
    default="fin_qa,wikitq,turl_col_type,tab_fact,numeric_nlg,qtsumm,tablebench_data_analysis,"
    "tablebench_fact_checking,tablebench_numerical_reasoning,tablebench_visualization",
)  # TODO: Scigen is dropped for now because it is implemented with 1 judge - make it 3?
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
    "shuffle_cols_names",
    "mask_cols_names",
    "shuffle_cols",
    "shuffle_rows",
    "insert_empty_rows",
    "duplicate_columns",
    "duplicate_rows",
    "transpose",
]
DEMOS_POOL_SIZE = 10

models_parsed = [item.strip() for item in models.split(",")]
cards_parsed = [item.strip() for item in cards.split(",")]
serializers_parsed = [item.strip() for item in serializers.split(",")]

augment_combinations = list(
    chain.from_iterable(
        combinations(TABLE_AUGMENTORS, r) for r in range(2, len(TABLE_AUGMENTORS) + 1)
    )
)
random.seed(settings.seed)
rand_augment_combinations = random.sample(
    augment_combinations, min(num_augmentors, len(augment_combinations))
)
"""
all augment includes:
    - no augment
    - one text augment
    - single table augments
    - some combinations of 2 augments or more
"""
all_augment = (
    [[None], ["augment_whitespace_task_input"]]
    + [[a] for a in TABLE_AUGMENTORS]
    + [list(i) for i in rand_augment_combinations]
)


# print("Run Params:", [f"{arg}: {value}" for arg, value in vars(args).items()])
for model in models_parsed:
    model_name = model.split("/")[-1]

    format = "formats.empty"
    if "llama" in model:
        format = "formats.llama3_instruct_all_demos_in_one_turn_without_system_prompt"
    elif "mixtral" in model:
        format = "formats.models.mistral.instruction.all_demos_in_one_turn"

    subsets = {}
    for card in cards_parsed:
        for augment in all_augment:
            for serializer in serializers_parsed:
                subset_name = (
                    "dataset="
                    + card
                    + "__serializer="
                    + serializer
                    + ("__augment=" + ",".join(augment) if augment != [None] else "")
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

    for subset_name, subset in tqdm(subsets.items()):
        # print("Running:", subset_name, ", Model:", model)

        benchmark = Benchmark(
            max_samples_per_subset=100 if not debug else 5,
            loader_limit=500 if not debug else 100,
            subsets={subset_name: subset},
        )

        test_dataset = list(benchmark()["test"])

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_file_name = (
            "model=" + model_name + "__" + subset_name + ("__DEBUG" if debug else "")
        )

        try:
            if "gpt" in model:
                inference_model = OpenAiInferenceEngine(
                    model_name=model,
                    max_tokens=max_pred_tokens,
                    temperature=0.05,
                )
            else:
                inference_model = IbmGenAiInferenceEngine(
                    model_name=model,
                    max_new_tokens=max_pred_tokens,
                    temperature=0.05,
                )

            predictions = inference_model.infer(test_dataset)
            evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

            curr_out_path = os.path.join(out_path, out_file_name) + ".pkl"
            with open(curr_out_path, "wb") as f:
                pickle.dump(evaluated_dataset, f)
                # print("saved file path: ", curr_out_path)
        except Exception as e:
            with open(os.path.join(out_path, "errors.txt"), "a") as f:
                f.write("\n".join([model, subset_name, str(e)]))
            # print(e)
            pass
