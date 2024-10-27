from tqdm import tqdm
from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import HFLlavaInferenceEngine
from unitxt.text_utils import print_dict
# from cvar_pyutils.debugging_tools import set_remote_debugger
# set_remote_debugger('9.61.73.90', 55557)
with settings.context(
    disable_hf_datasets_cache=False,
):
    inference_model = HFLlavaInferenceEngine(
        model_name="llava-hf/llava-interleave-qwen-0.5b-hf", max_new_tokens=32
    )

    dataset = load_dataset(
        card="cards.ai2d",
        template="templates.qa.multiple_choice.with_context.lmms_eval",
        format="formats.models.llava_interleave",
        # loader_limit=20,
        # augmentor="augmentors.image.grey_scale",
        augmentor="augmentors.image.rgb",
        streaming=True,
        metrics=["metrics.exact_match_mm"]
    )
    # dataset = load_dataset(
    #     card="cards.doc_vqa.lmms_eval",
    #     template="templates.qa.with_context.lmms_eval", # why do we need to define both the dataset and the template?
    #     format="formats.models.llava_interleave",
    #     loader_limit=20,
    #     # augmentor="augmentors.image.grey_scale",
    #     augmentor="augmentors.image.rgb",
    #     streaming=True,
    #     metrics=["metrics.anls"]
    # )
    # test_dataset = list(tqdm(dataset["test"], total=20))
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
