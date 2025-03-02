from unitxt import evaluate, load_dataset, settings
from unitxt.inference import (
    CrossProviderInferenceEngine, WMLInferenceEngineChat
)


with settings.context(
    disable_hf_datasets_cache=False,
    allow_unverified_code=True,
):
    test_dataset = load_dataset(
        "benchmarks.llama_vision[format=formats.chat_api,max_samples_per_subset=30]",
        split="test",
        use_cache=False
    )

# Infer
model = CrossProviderInferenceEngine(
    model="llama-3-2-11b-vision-instruct",
    max_tokens=32,
    provider="watsonx",
    temperature=0.0,
)
# model = WMLInferenceEngineChat(model_name="meta-llama/llama-3-2-11b-vision-instruct",
#                                max_tokens=32,
#                                temperature=0.0,
#                                )
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

"""
| subset   |    score | score_name      |   num_of_instances |
|:---------|---------:|:----------------|-------------------:|
| ALL      | 0.553122 | subsets_mean    |                120 |
| doc_vqa  | 0.666774 | anls            |                 30 |
| info_vqa | 0.51238  | anls            |                 30 |
| chart_qa | 0.266667 | relaxed_overall |                 30 |
| ai2d     | 0.766667 | exact_match_mm  |                 30 |
"""
