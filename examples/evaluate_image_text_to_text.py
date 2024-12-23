from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import HFLlavaInferenceEngine, LMMSEvalInferenceEngine, VLLMInferenceEngine
from unitxt.text_utils import print_dict
from tqdm import tqdm

from cvar_pyutils.debugging_tools import set_remote_debugger
# set_remote_debugger('9.61.188.58', 55557)

with settings.context(
    disable_hf_datasets_cache=False,
):
    # inference_model = HFLlavaInferenceEngine(
    #     model_name="llava-hf/llava-interleave-qwen-0.5b-hf", max_new_tokens=32
    # )
    # inference_model = LMMSEvalInferenceEngine(
    #     model_type="llava",
    #     model_args={"pretrained": "liuhaotian/llava-v1.5-7b"},
    #     max_new_tokens=32,
    # )
    inference_model = VLLMInferenceEngine(
        model="llava-hf/llava-1.5-7b-hf",
        max_tokens=32,
    )
    # dataset = load_dataset(
    #     card="cards.ai2d",
    #     template="templates.qa.multiple_choice.with_context.lmms_eval",
    #     format="formats.models.llava_interleave", Format should include the instruction from the dataset.
    #     # system_prompt="system_prompts.models.llava1_5", # need to insert this into the format
    #     # loader_limit=20,
    #     # augmentor="augmentors.image.grey_scale",
    #     augmentor="augmentors.image.to_rgb",
    #     streaming=True,
    #     metrics=["metrics.exact_match_mm"]
    # )
    dataset = load_dataset(
        card="cards.info_vqa", # docvqa.lmms_eval
        # template_card_index=0, # not needed in  newer version
        template="templates.qa.with_context.lmms_eval", # why do we need to define both the dataset and the template?
        format="formats.models.llava_interleave",
        loader_limit=20,
        # augmentor="augmentors.image.to_rgb",
        streaming=True,
        split="test",
        metrics=["metrics.anls"]
    )
    # dataset = load_dataset(
    #     card="cards.chart_qa",
    #     template="templates.qa.with_context.lmms_eval",
    #     format="formats.models.llava_interleave",
    #     # loader_limit=20,
    #     # augmentor="augmentors.image.grey_scale",
    #     augmentor="augmentors.image.to_rgb",
    #     streaming=True,
    #     metrics=["metrics.relaxed_correctness.json"]
    # )
    # test_dataset = list(tqdm(dataset["test"], total=20))
    # test_dataset = list(tqdm(dataset["test"]))

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
