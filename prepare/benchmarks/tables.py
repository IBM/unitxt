import argparse
import datetime
import os.path
import pickle

import torch
from tqdm import tqdm
from unitxt import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
    OpenAiInferenceEngine,
)
from unitxt.settings_utils import get_settings
from unitxt.standard import StandardRecipe
from unitxt.struct_data_operators import (
    SerializeTableAsDFLoader,
    SerializeTableAsIndexedRowMajor,
    SerializeTableAsJson,
    SerializeTableAsMarkdown,
)

settings = get_settings()

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", type=str, required=True)
parser.add_argument("-num_demos", "--num_demos", type=int, required=True)
parser.add_argument("-out_path", "--out_path", type=str, required=True)
parser.add_argument("-debug", "--debug", type=bool, default=False)
parser.add_argument("-seed", "--seed", type=int, required=False, default=settings.seed)
parser.add_argument(
    "-shuffle_rows", "--shuffle_rows", type=bool, required=False, default=False
)
parser.add_argument(
    "-shuffle_cols", "--shuffle_cols", type=bool, required=False, default=False
)
args = parser.parse_args()
model_name = args.model
num_demos = args.num_demos
out_path = args.out_path
debug = args.debug
seed = args.seed
shuffle_rows = args.shuffle_rows
shuffle_cols = args.shuffle_cols

# debug:
# python prepare/benchmarks/tables.py -model "google/flan-t5-base" -num_demos 1 -out_path "/Users/shir/Downloads/run_bench" -shuffle_rows True -debug True

# model_name = "google/flan-t5-base" #"gpt-4o-mini"
# num_demos = 0
# out_path = "/Users/shir/Downloads/run_bench"
# debug = False
# seed = settings.seed
# shuffle_rows = False
# shuffle_cols = False
# args = dict()
# print("debug ", debug)

DEMOS_POOL_SIZE = 10
datasets = ["tab_fact"]  # ["fin_qa", "wikitq", "tab_fact"]

all_subsets = {
    "fin_qa": {
        "fin_qa__json": StandardRecipe(
            card="cards.fin_qa",
            template_card_index=0,
            serializer=SerializeTableAsJson(shuffle_rows=shuffle_rows, seed=seed),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
        "fin_qa__markdown": StandardRecipe(
            card="cards.fin_qa",
            template_card_index=0,
            serializer=SerializeTableAsMarkdown(shuffle_rows=shuffle_rows, seed=seed),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
        "fin_qa__row_indexed_major": StandardRecipe(
            card="cards.fin_qa",
            template_card_index=0,
            serializer=SerializeTableAsIndexedRowMajor(
                shuffle_rows=shuffle_rows, seed=seed
            ),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
        "fin_qa__df": StandardRecipe(
            card="cards.fin_qa",
            template_card_index=0,
            serializer=SerializeTableAsDFLoader(shuffle_rows=shuffle_rows, seed=seed),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
    },
    "wikitq": {
        "wikitq__json": StandardRecipe(
            card="cards.wikitq",
            template_card_index=0,
            serializer=SerializeTableAsJson(shuffle_rows=shuffle_rows, seed=seed),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
        "wikitq__markdown": StandardRecipe(
            card="cards.wikitq",
            template_card_index=0,
            serializer=SerializeTableAsMarkdown(shuffle_rows=shuffle_rows, seed=seed),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
        "wikitq__row_indexed_major": StandardRecipe(
            card="cards.wikitq",
            template_card_index=0,
            serializer=SerializeTableAsIndexedRowMajor(
                shuffle_rows=shuffle_rows, seed=seed
            ),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
        "wikitq__df": StandardRecipe(
            card="cards.wikitq",
            template_card_index=0,
            serializer=SerializeTableAsDFLoader(shuffle_rows=shuffle_rows, seed=seed),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
    },
    "tab_fact": {
        "tab_fact__json": StandardRecipe(
            card="cards.tab_fact",
            template_card_index=0,
            serializer=SerializeTableAsJson(shuffle_rows=shuffle_rows, seed=seed),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
        "tab_fact__markdown": StandardRecipe(
            card="cards.tab_fact",
            template_card_index=0,
            serializer=SerializeTableAsMarkdown(shuffle_rows=shuffle_rows, seed=seed),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
        "tab_fact__row_indexed_major": StandardRecipe(
            card="cards.tab_fact",
            template_card_index=0,
            serializer=SerializeTableAsIndexedRowMajor(
                shuffle_rows=shuffle_rows, seed=seed
            ),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
        "tab_fact__df": StandardRecipe(
            card="cards.tab_fact",
            template_card_index=0,
            serializer=SerializeTableAsDFLoader(shuffle_rows=shuffle_rows, seed=seed),
            num_demos=num_demos,
            demos_pool_size=DEMOS_POOL_SIZE,
        ),
    },
}

subsets = {key: vals for key, vals in all_subsets.items() if key in datasets}

for subset_name, subset in tqdm(subsets.items()):
    # print("Running:", subset_name, "|", [f"{arg}: {value} | " for arg, value in vars(args).items()],)

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
                temperature=0.001,
            )
        else:
            inference_model = HFPipelineBasedInferenceEngine(
                model_name=model_name,
                max_new_tokens=100,
                use_fp16=True,
                # temperature=0 is hard coded in HFPipelineBasedInferenceEngine since it is not allowed to be a param
            )

        predictions = inference_model.infer(test_dataset)
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

        # exp_name = "-{0}{1}{2}".format(("shuffle_cols" if shuffle_cols else ""),("shuffle_rows" if shuffle_rows else ""),(seed if seed != settings.seed else ""),)

        out_file_name = (
            model_name.replace("/", "_")
            + "#"
            + subset_name
            # + (exp_name if exp_name else "")
            + "#"
            + str(datetime.datetime.now())
        )
        curr_out_path = os.path.join(out_path, out_file_name) + ".pkl"
        with open(curr_out_path, "wb") as f:
            pickle.dump(evaluated_dataset, f)
            # print("saved file path: ", curr_out_path)
    except Exception:
        # print(e)
        pass
