from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import HFLlavaInferenceEngine
from unitxt.text_utils import print_dict

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

    inference_model = HFLlavaInferenceEngine(
        model_name="llava-hf/llava-interleave-qwen-0.5b-hf", max_new_tokens=32
    )

    predictions = inference_model.infer(dataset)
    evaluated_dataset = evaluate(predictions=predictions, data=dataset)

    print_dict(
        evaluated_dataset[0],
        keys_to_print=[
            "source",
            "media",
            "references",
            "processed_prediction",
            "score",
        ],
    )
