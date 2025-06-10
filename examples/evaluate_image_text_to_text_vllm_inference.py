from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    VLLMInferenceEngine,
)

with settings.context(
    disable_hf_datasets_cache=False,
):
    max_tokens = 512
    dataset = load_dataset(
        card="cards.chart_qa_lmms_eval",
        format="formats.chat_api",
        loader_limit=30,
        split="test",
    )

    inference_model = VLLMInferenceEngine(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        max_tokens=max_tokens,
        temperature=0.0,
    )

    predictions = inference_model(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
