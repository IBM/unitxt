import datetime
import os.path
import pickle
import random

import torch
from unitxt import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.standard import StandardRecipe
from unitxt.struct_data_operators import (
    SerializeTableAsDFLoader,
    SerializeTableAsIndexedRowMajor,
    SerializeTableAsJson,
    SerializeTableAsMarkdown,
)

debug = False if os.path.exists("/dccstor") else True

subsets = set()
SEED_NUM_OF_RUNS = 10

for _ in range(SEED_NUM_OF_RUNS):
    rand = random.randint(0, 5000)
    subsets = subsets.update(
        {
            "json": StandardRecipe(
                card="cards.fin_qa",
                template_card_index=0,
                serializer=SerializeTableAsJson(shuffle_rows=True, seed=rand),
            ),
            "markdown": StandardRecipe(
                card="cards.fin_qa",
                template_card_index=0,
                serializer=SerializeTableAsMarkdown(shuffle_rows=True, seed=rand),
            ),
            "row_indexed_major": StandardRecipe(
                card="cards.fin_qa",
                template_card_index=0,
                serializer=SerializeTableAsIndexedRowMajor(
                    shuffle_rows=True, seed=rand
                ),
            ),
            "df": StandardRecipe(
                card="cards.fin_qa",
                template_card_index=0,
                serializer=SerializeTableAsDFLoader(shuffle_rows=True, seed=rand),
            ),
        }
    )


benchmark = Benchmark(
    max_samples_per_subset=100 if not debug else 5,
    loader_limit=3000 if not debug else 100,
    subsets=subsets,
)

test_dataset = list(benchmark()["test"])

if not debug:
    models = ["Qwen/Qwen2.5-Math-72B-Instruct", "mistralai/Mixtral-8x22B-Instruct-v0.1"]
    out_path = "/dccstor/gmc/shir/data_comp/run_bench"
else:
    models = ["google/flan-t5-base", "google/flan-t5-large"]
    out_path = "/Users/shir/Downloads/run_bench"

if not os.path.exists(out_path):
    os.makedirs(out_path)

for model_name in models:
    try:
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        inference_model = HFPipelineBasedInferenceEngine(
            model_name=model_name,
            max_new_tokens=32,
        )
        predictions = inference_model.infer(test_dataset)
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

        out_file_name = (
            model_name.replace("/", "_") + "#" + str(datetime.datetime.now())
        )
        with open(os.path.join(out_path, out_file_name) + ".pkl", "wb") as f:
            pickle.dump(evaluated_dataset, f)
    except:
        pass
