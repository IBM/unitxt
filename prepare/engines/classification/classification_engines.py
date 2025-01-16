from unitxt import add_to_catalog
from unitxt.inference import (
    AzureOpenAIInferenceEngine,
    CrossProviderInferenceEngine,
    WMLInferenceEngineGeneration,
)

model_names_to_provider = {
    "llama-3-3-70b-instruct": ["watsonx", "rits"],
    "llama-3-1-70b-instruct": ["watsonx", "rits"],
    "gpt-4o": ["open-ai"],
    "gpt-4-turbo": ["open-ai"],
    "gpt-4-turbo-2024-04-09": ["azure"],
    "gpt-4o-2024-08-06": ["azure"],
    "mistralai/mixtral-8x7b-instruct-v01": ["ibm_wml"],
    "meta-llama/llama-3-3-70b-instruct": ["ibm_wml"],
    "meta-llama/llama-3-1-70b-instruct": ["ibm_wml"],
    "meta-llama/llama-3-405b-instruct": ["ibm_wml"],
    "llama-3-1-405b-instruct-fp8": ["rits"],
}


def get_inference_engine(model_name, provider):
    if provider == "ibm_wml":
        return WMLInferenceEngineGeneration(
            model_name=model_name,
            max_new_tokens=5,
            random_seed=42,
            decoding_method="greedy",
        )

    if provider == "azure":
        return AzureOpenAIInferenceEngine(
            model_name=model_name,
            logprobs=True,
            max_tokens=5,
            temperature=0.0,
            top_logprobs=5,
        )

    return CrossProviderInferenceEngine(
        model=model_name,
        logprobs=True,
        max_tokens=5,
        temperature=0.0,
        top_logprobs=5,
        provider=provider,
    )


for judge_model_name, infer_frameworks in model_names_to_provider.items():
    for infer_framework in infer_frameworks:
        inference_engine = get_inference_engine(judge_model_name, infer_framework)
        inference_engine_label = inference_engine.get_engine_id().replace("-", "_")

        add_to_catalog(
            inference_engine,
            f"engines.classification.{inference_engine_label}",
            overwrite=True,
        )
