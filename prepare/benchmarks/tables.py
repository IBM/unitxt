import datetime
import os.path
import pickle

import torch
from unitxt import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.standard import StandardRecipe
from unitxt.struct_data_operators import (
    SerializeTableAsJson,
)

debug = False if os.path.exists("/dccstor") else True


subsets = {
    # "fin_qa": Benchmark(
    #     max_samples_per_subset=100 if not debug else 5,
    #     subsets={
    #         "json": StandardRecipe(
    #             card="cards.fin_qa",
    #             template_card_index=0,
    #             serializer=SerializeTableAsJson(),
    #         ),
    # "markdown": StandardRecipe(card="cards.fin_qa", template_card_index=0,
    #                            serializer=SerializeTableAsMarkdown()),
    # "row_indexed_major": StandardRecipe(card="cards.fin_qa", template_card_index=0,
    #                                     serializer=SerializeTableAsIndexedRowMajor()),
    # "df": StandardRecipe(card="cards.fin_qa", template_card_index=0,
    #                      serializer=SerializeTableAsDFLoader()),
    #     },
    # ),
    "wikitq": Benchmark(
        max_samples_per_subset=100 if not debug else 5,
        subsets={
            "json": StandardRecipe(
                card="cards.wikitq",
                template_card_index=0,
                serializer=SerializeTableAsJson(),
            ),
            #         "markdown": StandardRecipe(card="cards.wikitq", template_card_index=0,
            #                                    serializer=SerializeTableAsMarkdown()),
            #         "row_indexed_major": StandardRecipe(card="cards.wikitq", template_card_index=0,
            #                                             serializer=SerializeTableAsIndexedRowMajor()),
            #         "df": StandardRecipe(card="cards.wikitq", template_card_index=0,
            #                              serializer=SerializeTableAsDFLoader()),
        },
    )
}

benchmark = Benchmark(
    max_samples_per_subset=100 if not debug else 5,
    loader_limit=3000 if not debug else 100,
    subsets={
        "fin_qa": Benchmark(
            max_samples_per_subset=100 if not debug else 5,
            subsets=subsets,
        )
    },
)

test_dataset = list(benchmark()["test"])

if not debug:
    models = ["Qwen/Qwen2.5-72B-Instruct", "mistralai/Mixtral-8x22B-Instruct-v0.1"]
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
            use_fp16=True,
        )
        predictions = inference_model.infer(test_dataset)
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

        out_file_name = (
            model_name.replace("/", "_") + "#" + str(datetime.datetime.now())
        )
        curr_out_path = os.path.join(out_path, out_file_name) + ".pkl"
        with open(curr_out_path, "wb") as f:
            pickle.dump(evaluated_dataset, f)
            # print("saved file path: ", curr_out_path)
    except Exception:
        # print(e)
        pass
