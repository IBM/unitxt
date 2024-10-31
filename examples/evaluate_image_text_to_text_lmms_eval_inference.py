from tqdm import tqdm
from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    LMMSEvalInferenceEngine,
)
from unitxt.text_utils import print_dict

with settings.context(
    disable_hf_datasets_cache=False,
):
    inference_model = LMMSEvalInferenceEngine(
        model_type="llava_onevision",
        model_args={"pretrained": "lmms-lab/llava-onevision-qwen2-7b-ov"},
        max_new_tokens=2,
    )

    dataset = load_dataset(
        card="cards.seed_bench",
        template="templates.qa.multiple_choice.with_context.lmms_eval",
        # loader_limit=30,
        streaming=True,
    )

    test_dataset = list(tqdm(dataset["test"], total=30))

    predictions = inference_model.infer(test_dataset)
    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

    print_dict(
        evaluated_dataset[7],
        keys_to_print=[
            "source",
            "media",
            "references",
            "processed_prediction",
            "score",
        ],
    )
