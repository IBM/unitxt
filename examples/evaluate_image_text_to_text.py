from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import LMMSEvalInferenceEngine

with settings.context(disable_hf_datasets_cache=False):
    inference_model = LMMSEvalInferenceEngine(
        model_type="llama_vision",
        model_args={"pretrained": "meta-llama/Llama-3.2-11B-Vision-Instruct"},
        # model_type="llava",
        # model_args={"pretrained": "liuhaotian/llava-v1.5-7b"},
        max_new_tokens=32,
        image_token="",
    )

    dataset = load_dataset(
        card="cards.chart_qa_lmms_eval",
        format="formats.chat_api",
        split="test",
        max_test_instances=20,
        # disable_cache=True
    )

    predictions = inference_model.infer(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
