from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

with settings.context(
    disable_hf_datasets_cache=False,
):
    model = CrossProviderInferenceEngine(
        model="llama-3-3-70b-instruct", provider="watsonx"
    )
    dataset = load_dataset(
        card="cards.coqa.multi_turn",
        format="formats.chat_api",
        split="test",
        max_test_instances=100,
    )

    predictions = model.infer(dataset)

    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)
