from unitxt import evaluate, load_dataset
from unitxt.inference import MultiAPIInferenceEngine
from unitxt.text_utils import print_dict

data = load_dataset(
    "benchmarks.glue[max_samples_per_subset=5, format=formats.chat_api]",
    split="test",
    disable_cache=False,
)

model = MultiAPIInferenceEngine(model="llama-3-8b-instruct", top_k=1, api="watsonx")

predictions = model.infer(data)

evaluated_dataset = evaluate(predictions=predictions, data=data)

print_dict(
    evaluated_dataset[0],
    keys_to_print=[
        "source",
        "prediction",
        "subset",
    ],
)
print_dict(
    evaluated_dataset[0]["score"]["subsets"],
)
