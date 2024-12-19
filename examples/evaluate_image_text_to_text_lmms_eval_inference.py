from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    LMMSEvalInferenceEngine,
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

    model = LMMSEvalInferenceEngine(
        model_type="llava",
        model_args={
            "pretrained": "liuhaotian/llava-v1.5-7b",
        },
        max_new_tokens=2,
    )

    predictions = model(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
