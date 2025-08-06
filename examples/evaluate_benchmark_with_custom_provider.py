from unitxt import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine

dataset = load_dataset(
    "benchmarks.glue",
    format="formats.chat_api",
    system_prompt="system_prompts.general.be_concise",
    max_samples_per_subset=5,
    split="test",
    use_cache=True,
)

model = CrossProviderInferenceEngine(
    model="llama-3-2-3b-instruct", temperature=0.0, top_p=1.0, provider="watsonx"
)
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""

predictions = model(dataset)

results = evaluate(predictions=predictions, data=dataset)

print("Global Results:")
print(results.global_scores.summary)

print("Subsets Results:")
print(results.subsets_scores.summary)
