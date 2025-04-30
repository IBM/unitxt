import os.path
import pickle

from unitxt import evaluate, load_dataset, settings
from unitxt.inference import HFAutoModelInferenceEngine

out_path = "results_log_probs"
model_name = "meta-llama/Llama-3.1-8B-Instruct"

with settings.context(
    allow_unverified_code=True,
):
    test_dataset = load_dataset(
        # "benchmarks.torr[loader_limit=100,max_samples_per_subset=8]",
        card="cards.fin_qa",
        split="test",
        use_cache=True,
        loader_limit=100,
    )

    model = HFAutoModelInferenceEngine(
        model_name=model_name,
        max_new_tokens=64,
        n_top_tokens=100,
        batch_size=8,
        do_sample=True,
    )

    predictions = model.infer_log_probs(test_dataset, return_meta_data=True)
    # print(predictions)
    results = evaluate(predictions=predictions, data=test_dataset)
    # print(predictions)

    out_file_name = (
            "model="
            + model_name.replace("/", "_")
            + ".pkl"
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(os.path.join(out_path, out_file_name, ",file=predict"), "wb") as f:
        pickle.dump(predictions, f)

    with open(os.path.join(out_path, out_file_name, ",file=eval"), "wb") as f:
        pickle.dump(results, f)
