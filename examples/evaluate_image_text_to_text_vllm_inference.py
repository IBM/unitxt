from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    VLLMInferenceEngine,
)
from unitxt.text_utils import print_dict

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

    predictions = inference_model.infer(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print_dict(results[0], keys_to_print=["prediction", "target", "references"])
    print_dict(
        results[0]["score"]["global"],
    )
