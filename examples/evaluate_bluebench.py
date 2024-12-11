from unitxt import evaluate, load_dataset, settings
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.text_utils import print_dict

with settings.context(
    disable_hf_datasets_cache=False,
    allow_unverified_code=True,
):
    test_dataset = load_dataset("benchmarks.bluebench", split="test")

# Infer
inference_model = CrossProviderInferenceEngine(
    model="llama-3-8b-instruct",
    max_tokens=30,
)
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""

predictions = inference_model.infer(test_dataset)
results = evaluate(predictions=predictions, data=test_dataset)

print_dict(
    results.subsets_scores,
)
