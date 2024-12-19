from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    VLLMInferenceEngine,
)

with settings.context(
    disable_hf_datasets_cache=False,
):
    dataset = load_dataset(
        card="cards.seed_bench",
        format="formats.chat_api",
        loader_limit=30,
        split="test",
    )

    inference_model = VLLMInferenceEngine(
        model="microsoft/Phi-3-vision-128k-instruct",
        max_tokens=2,
    )

    predictions = inference_model(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
