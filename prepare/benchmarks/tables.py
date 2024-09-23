import datetime
import os.path
import pickle

from unitxt import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.standard import StandardRecipe

debug = False if os.path.exists("/dccstor") else True

benchmark = Benchmark(
    max_samples_per_subset=100,
    loader_limit=3000,
    subsets={
        # "fin_qa__concat_text": StandardRecipe(card="cards.fin_qa__concat_text", template_card_index=0),
        # "fin_qa__df": StandardRecipe(card="cards.fin_qa__df", template_card_index=0),
        # "fin_qa__indexedrowmajor": StandardRecipe(card="cards.fin_qa__indexedrowmajor", template_card_index=0),
        # "fin_qa__json": StandardRecipe(card="cards.fin_qa__json", template_card_index=0),
        # "fin_qa__markdown": StandardRecipe(card="cards.fin_qa__markdown", template_card_index=0),
        # "fin_qa": StandardRecipe(card="cards.fin_qa", template_card_index=0),
        "wikitq__concat_text": StandardRecipe(
            card="cards.wikitq__concat_text", template_card_index=0
        ),
        "wikitq__df": StandardRecipe(card="cards.wikitq__df", template_card_index=0),
        "wikitq__df__shuffle_rows": StandardRecipe(
            card="cards.wikitq__df__shuffle_rows", template_card_index=0
        ),
        "wikitq__indexed_row_major": StandardRecipe(
            card="cards.wikitq__indexed_row_major", template_card_index=0
        ),
        "wikitq__indexed_row_major__shuffle_rows": StandardRecipe(
            card="cards.wikitq__indexed_row_major__shuffle_rows", template_card_index=0
        ),
        "wikitq__json": StandardRecipe(
            card="cards.wikitq__json", template_card_index=0
        ),
        "wikitq__json__empty_table": StandardRecipe(
            card="cards.wikitq__json__empty_table", template_card_index=0
        ),
        "wikitq__json__junk_table": StandardRecipe(
            card="cards.wikitq__json__junk_table", template_card_index=0
        ),
        "wikitq__markdown": StandardRecipe(
            card="cards.wikitq__markdown", template_card_index=0
        ),
        "wikitq__markdown__shuffle_cols": StandardRecipe(
            card="cards.wikitq__markdown", template_card_index=0
        ),
    },
)

test_dataset = list(benchmark()["test"])

if not debug:
    models = ["Qwen/Qwen2.5-Math-72B-Instruct", "mistralai/Mixtral-8x22B-Instruct-v0.1"]
    out_path = "/dccstor/gmc/shir/data_comp/tables_wiki"
else:
    models = ["google/flan-t5-base", "google/flan-t5-large"]
    out_path = "/Users/shir/Downloads/tables_wiki"

for model_name in models:
    inference_model = HFPipelineBasedInferenceEngine(
        model_name=model_name, max_new_tokens=32
    )
    predictions = inference_model.infer(test_dataset)
    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

    out_file_name = model_name.replace("/", "_") + "#" + str(datetime.datetime.now())
    with open(os.path.join(out_path, out_file_name) + ".pkl", "wb") as f:
        pickle.dump(evaluated_dataset, f)
