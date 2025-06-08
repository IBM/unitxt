import os

from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine

os.environ["WML_URL"] = "https://us-south.ml.cloud.ibm.com"
os.environ["WML_APIKEY"] = ""
os.environ["WML_PROJECT_ID"] = ""
os.environ["WATSONX_URL"] = "https://us-south.ml.cloud.ibm.com"
os.environ["WATSONX_API_KEY"] = ""
os.environ["WATSONX_PROJECT_ID"] = ""
# logging
if False:
    os.environ["LITELLM_LOG"] = "DEBUG"
    import logging

    logging.basicConfig(level=logging.DEBUG)
    # additional settings for clean logs
    httpcore_logging = logging.getLogger("httpcore")
    httpcore_logging.setLevel(logging.ERROR)
    httpx_logging = logging.getLogger("httpx")
    httpx_logging.setLevel(logging.ERROR)
with settings.context(disable_hf_datasets_cache=False):
    max_tokens = 512
    dataset = load_dataset(
        card="cards.ai2d",
        format="formats.chat_api",
        split="test",
        max_test_instances=20,
    )

    inference_model = CrossProviderInferenceEngine(
        model="llama-3-2-11b-vision-instruct",
        provider="watsonx",
        max_tokens=max_tokens,
        temperature=0.0,
    )
    # inference_model = WMLInferenceEngineChat(model_name="meta-llama/llama-3-2-11b-vision-instruct",
    #                                          max_tokens=max_tokens, temperature=0.0)

    predictions = inference_model.infer(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
