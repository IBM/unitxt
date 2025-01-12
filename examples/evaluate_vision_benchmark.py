from unitxt import evaluate, load_dataset, settings
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

with settings.context(
    disable_hf_datasets_cache=False,
    allow_unverified_code=True,
):
    test_dataset = load_dataset(
        "benchmarks.vision[loader_limit=30,max_samples_per_subset=30]", split="test"
    )

# Infer
model = CrossProviderInferenceEngine(
    model="llama-3-2-11b-vision-instruct",
    max_tokens=30,
)
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""

predictions = model(test_dataset)
results = evaluate(predictions=predictions, data=test_dataset)

print("Global scores:")
print(results.global_scores.summary)
print("Subsets scores:")
print(results.subsets_scores.summary)

# | subset   |    score | score_name      |   num_of_instances |
# |:---------|---------:|:----------------|-------------------:|
# | ALL      | 0.15637  | subsets_mean    |                150 |
# | doc_vqa  | 0.05     | anls            |                 30 |
# | info_vqa | 0.176852 | anls            |                 30 |
# | chart_qa | 0.1      | relaxed_overall |                 30 |
# | ai2d     | 0.3      | exact_match_mm  |                 30 |
# | websrc   | 0.155    | websrc_squad_f1 |                 30 |
