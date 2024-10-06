from tqdm import tqdm
from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import HFLlavaInferenceEngine
from unitxt.text_utils import print_dict

with settings.context(
    disable_hf_datasets_cache=False,
):
    inference_model = HFLlavaInferenceEngine(
        model_name="llava-hf/llava-interleave-qwen-0.5b-hf", max_new_tokens=32
    )

    dataset = load_dataset(
        card="cards.doc_vqa.lmms_eval",
        template="templates.qa.with_context.title",
        format="formats.models.llava_interleave",
        loader_limit=300,
        augmentor="augmentors.image.grey_scale",
        streaming=True,
    )

    test_dataset = list(tqdm(dataset["test"]))

    predictions = inference_model.infer(test_dataset)
    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

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
