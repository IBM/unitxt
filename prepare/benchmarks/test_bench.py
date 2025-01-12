import os

from unitxt import evaluate, load_dataset, settings
from unitxt.inference import CrossProviderInferenceEngine

# default params
MAX_PREDICTIONS = 100
LOADER_LIMIT = 10000
MAX_PRED_TOKENS = 100
TEMPERATURE = 0.05
DESCRIPTIVE_DATASETS = {
    "scigen",
    "numeric_nlg",
    "qtsumm",
    "tablebench_visualization",
    "tablebench_data_analysis",
}
out_path = "debug"
models = "meta-llama/llama-3-1-70b-instruct"
debug = True
out_file_name = "tables_benchmark_results.csv"

# parse params
models_parsed = [item.strip() for item in models.split(",")]
if not os.path.exists(out_path):
    os.makedirs(out_path)


# run benchmark
with settings.context(
    disable_hf_datasets_cache=False,
):
    benchmark = load_dataset(
        "benchmarks.tables_benchmark[max_samples_per_subset=5,loader_limit=50]"
    )
    test_dataset = benchmark()["test"].to_dataset()

    for model in models_parsed:
        model_name = model.split("/")[-1]

        inference_model = CrossProviderInferenceEngine(
            model=model_name,
            max_tokens=30,
            provider="rits",
        )

        predictions = inference_model.infer(test_dataset)
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)
        evaluated_dataset.instance_scores.to_df().to_csv(out_file_name)
