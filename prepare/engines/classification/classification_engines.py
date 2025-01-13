from unitxt import add_to_catalog
from unitxt.inference import (
    OpenAiInferenceEngine,
    RITSInferenceEngine,
    WMLInferenceEngineGeneration,
)


def get_inference_engine(model_name, framework_name):
    if framework_name == "ibm_wml":
        return WMLInferenceEngineGeneration(
            model_name=model_name,
            max_new_tokens=5,
            random_seed=42,
            decoding_method="greedy",
        )

    if framework_name == "openai":
        return OpenAiInferenceEngine(
            model_name=model_name, logprobs=True, max_tokens=5, temperature=0.0
        )
    if framework_name == "rits":
        return RITSInferenceEngine(
            model_name=model_name, logprobs=True, max_tokens=5, temperature=0.0
        )
    raise ValueError("Unsupported framework name " + framework_name)


model_names_to_infer_framework = {
    "meta-llama/llama-3-1-70b-instruct": ["ibm_wml", "rits"],
    "gpt-4-turbo": ["openai"],
    "mistralai/mixtral-8x7b-instruct-v01": ["ibm_wml", "rits"],
    "meta-llama/llama-3-1-405b-instruct-fp8": ["rits"],
    "meta-llama/llama-3-405b-instruct": ["ibm_wml"],
}

for judge_model_name, infer_frameworks in model_names_to_infer_framework.items():
    for infer_framework in infer_frameworks:
        inference_engine = get_inference_engine(judge_model_name, infer_framework)
        inference_engine_label = inference_engine.get_engine_id()

        add_to_catalog(
            inference_engine,
            f"engines.classification.{inference_engine_label}",
            overwrite=True,
        )
