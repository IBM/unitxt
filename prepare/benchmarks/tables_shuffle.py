import argparse
import datetime
import os.path
import pickle

import torch
from unitxt import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
    OpenAiInferenceEngine,
)
from unitxt.standard import StandardRecipe
from unitxt.struct_data_operators import (
    SerializeTableAsDFLoader,
    SerializeTableAsIndexedRowMajor,
    SerializeTableAsJson,
    SerializeTableAsMarkdown,
)

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", type=str, required=True)
parser.add_argument("-num_demos", "--num_demos", type=int, required=True)
parser.add_argument("-debug", "--debug", type=bool, default=False)
parser.add_argument("-out_path", "--out_path", type=str, required=True)
args = parser.parse_args()
model_name = args.model
num_demos = args.num_demos
out_path = args.out_path
debug = args.debug

# model_name = "google/flan-t5-large" #"gpt-4o-mini"
# num_demos = 0
# out_path = "/Users/shir/Downloads/run_bench"
# debug = False
# print("debug ", debug)

DEMOS_POOL_SIZE = 10
# seed = random.randint(0, 3000)
seed = 564
subsets = {
    "fin_qa__json_shuffle-rows-564": StandardRecipe(
        card="cards.fin_qa",
        template_card_index=0,
        serializer=SerializeTableAsJson(shuffle_rows=True, seed=seed),
        num_demos=num_demos,
        demos_pool_size=DEMOS_POOL_SIZE,
    ),
    "fin_qa__markdown_shuffle-rows-564": StandardRecipe(
        card="cards.fin_qa",
        template_card_index=0,
        serializer=SerializeTableAsMarkdown(shuffle_rows=True, seed=seed),
        num_demos=num_demos,
        demos_pool_size=DEMOS_POOL_SIZE,
    ),
    "fin_qa__row_indexed_major_shuffle-rows-564": StandardRecipe(
        card="cards.fin_qa",
        template_card_index=0,
        serializer=SerializeTableAsIndexedRowMajor(shuffle_rows=True, seed=seed),
        num_demos=num_demos,
        demos_pool_size=DEMOS_POOL_SIZE,
    ),
    "fin_qa__df_shuffle-rows-564": StandardRecipe(
        card="cards.fin_qa",
        template_card_index=0,
        serializer=SerializeTableAsDFLoader(shuffle_rows=True, seed=seed),
        num_demos=num_demos,
        demos_pool_size=DEMOS_POOL_SIZE,
    ),
    "wikitq__json_shuffle-rows-564": StandardRecipe(
        card="cards.wikitq",
        template_card_index=0,
        serializer=SerializeTableAsJson(shuffle_rows=True, seed=seed),
        num_demos=num_demos,
        demos_pool_size=DEMOS_POOL_SIZE,
    ),
    "wikitq__markdown_shuffle-rows-564": StandardRecipe(
        card="cards.wikitq",
        template_card_index=0,
        serializer=SerializeTableAsMarkdown(shuffle_rows=True, seed=seed),
        num_demos=num_demos,
        demos_pool_size=DEMOS_POOL_SIZE,
    ),
    "wikitq__row_indexed_major_shuffle-rows-564": StandardRecipe(
        card="cards.wikitq",
        template_card_index=0,
        serializer=SerializeTableAsIndexedRowMajor(),
        num_demos=num_demos,
        demos_pool_size=DEMOS_POOL_SIZE,
    ),
    "wikitq__df_shuffle-rows-564": StandardRecipe(
        card="cards.wikitq",
        template_card_index=0,
        serializer=SerializeTableAsDFLoader(),
        num_demos=num_demos,
        demos_pool_size=DEMOS_POOL_SIZE,
    ),
}

for subset_name, subset in subsets.items():
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

        out_file_name = (
            model_name.replace("/", "_")
            + "#"
            + subset_name
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
