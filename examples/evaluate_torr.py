from unitxt import evaluate, load_dataset, settings
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

with settings.context(
    allow_unverified_code=True,
):
    test_dataset = load_dataset(
        "benchmarks.torr[max_samples_per_subset=100]",
        split="test",
        use_cache=True,
    )
    # Infer
    model = CrossProviderInferenceEngine(   # We used Together AI as inference engine
        model="llama-3-8b-instruct",
        max_tokens=512,
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
