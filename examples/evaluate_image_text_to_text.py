from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import HFLlavaInferenceEngine

with settings.context(
    disable_hf_datasets_cache=False,
):
    dataset = load_dataset(
        card="cards.doc_vqa.lmms_eval",
        template="templates.qa.with_context.title",
        format="formats.chat_api",
        loader_limit=10,
        augmentor="augmentors.image.grey_scale",
        split="test",
    )

    model = HFLlavaInferenceEngine(
        model_name="llava-hf/llava-interleave-qwen-0.5b-hf", max_new_tokens=32
    )

    predictions = model(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
