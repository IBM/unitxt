from unitxt import evaluate, load_dataset, settings
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.text_utils import print_dict

with settings.context(
    disable_hf_datasets_cache=True,
    allow_unverified_code=True,
):
    test_dataset = load_dataset(
        "card=cards.text2sql.bird,template=templates.text2sql.you_are_given_with_hint",
        split="validation",
    )

# Infer
inference_model = CrossProviderInferenceEngine(
    model="llama-3-70b-instruct",
    max_tokens=256,
)
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""

predictions = inference_model.infer(test_dataset)
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

print_dict(
    evaluated_dataset[0],
    keys_to_print=[
        "source",
        "prediction",
        "subset",
    ],
)
print_dict(
    evaluated_dataset[0]["score"]["global"],
)

# num_of_instances (int):
#     1534
# execution_accuracy (float):
#     0.46870925684485004

# like GPT4 (rank 40 in the benchmark https://bird-bench.github.io/)
