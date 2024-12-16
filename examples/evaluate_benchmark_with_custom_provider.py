from unitxt import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine

data = load_dataset(
    "benchmarks.glue[max_samples_per_subset=5, format=formats.chat_api, system_prompt=system_prompts.general.be_concise]",
    split="test",
    disable_cache=False,
)

model = CrossProviderInferenceEngine(
    model="llama-3-8b-instruct", temperature=0.0, top_p=1.0, provider="watsonx"
)
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""

predictions = model(data)

results = evaluate(predictions=predictions, data=data)

print(
    results.instance_scores.to_df(
        columns=[
            "source",
            "prediction",
            "subset",
        ]
    )
)

print(
    results.subsets_scores,
)
