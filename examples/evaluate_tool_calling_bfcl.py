from unitxt import evaluate, load_dataset, settings
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

with settings.context(
    allow_unverified_code=True,
):
    dataset = load_dataset(
        card="cards.bfcl.multi_turn.simple_v3",
        split="test",
        format="formats.chat_api",
        max_test_instances=10,
    )
    model = CrossProviderInferenceEngine(
        model="llama-3-1-8b-instruct", provider="watsonx"
    )

    predictions = model(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global scores:")
    print(results.global_scores.summary)
