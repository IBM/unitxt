from unitxt import evaluate, load_dataset, settings
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

with settings.context(
    disable_hf_datasets_cache=False,
    allow_unverified_code=True,
):
    test_dataset = load_dataset(
        "benchmarks.vision_full[format=formats.chat_api,max_samples_per_subset=512]",
        split="test",
    )

# Infer
model = CrossProviderInferenceEngine(
    model="llama-3-2-11b-vision-instruct",
    max_tokens=32,
    provider="watsonx",
    temperature=0.0,
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
"""
Using LLAMA-VISION-11B
| subset                         |    score | score_name      |   num_of_instances |
|:-------------------------------|---------:|:----------------|-------------------:|
| ALL                            | 0.462355 | subsets_mean    |               4608 |
| doc_vqa_default                | 0.7814   | anls            |                512 |
| info_vqa_default               | 0.562389 | anls            |                512 |
| chart_qa_default               | 0.197266 | relaxed_overall |                512 |
| ai2d_default                   | 0.126953 | exact_match_mm  |                512 |
| websrc_default                 | 0.371    | websrc_squad_f1 |                512 |
| doc_vqa_llama_vision_template  | 0.653235 | anls            |                512 |
| info_vqa_llama_vision_template | 0.508014 | anls            |                512 |
| chart_qa_llama_vision_template | 0.197266 | relaxed_overall |                512 |
| ai2d_llama_vision_template     | 0.763672 | exact_match_mm  |                512 |

"""
