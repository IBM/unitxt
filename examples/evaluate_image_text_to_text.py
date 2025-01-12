from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    LMMSEvalInferenceEngine,
)

with settings.context(
    disable_hf_datasets_cache=False,
):
    inference_model = LMMSEvalInferenceEngine(
        model_type="llava",
        model_args={"pretrained": "liuhaotian/llava-v1.5-7b"},
        max_new_tokens=128,
    )
    dataset = load_dataset(
        card="cards.websrc",
        format="formats.chat_api",
        # max_test_instances=20,
        split="test",
    )

    predictions = inference_model.infer(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
