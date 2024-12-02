from unitxt.catalog import add_to_catalog
from unitxt.inference import (
    MockInferenceEngine,
    OllamaInferenceEngine,
    IbmGenAiInferenceEngine,
    OpenAiInferenceEngine,
    VLLMRemoteInferenceEngine,
    RITSInferenceEngine,
    TogetherAiInferenceEngine,
    WMLInferenceEngineGeneration,
    WMLInferenceEngineChat,
    LiteLLMInferenceEngine,
    CrossProviderInferenceEngine
)

for klass in [
    OpenAiInferenceEngine, MockInferenceEngine, OllamaInferenceEngine,
    IbmGenAiInferenceEngine, OpenAiInferenceEngine, VLLMRemoteInferenceEngine,
    RITSInferenceEngine, TogetherAiInferenceEngine, WMLInferenceEngineGeneration,
    WMLInferenceEngineChat, LiteLLMInferenceEngine, CrossProviderInferenceEngine
]:
    try:
        inference_engine = klass(model_name=None)
    except:
        inference_engine = klass(model=None)

    add_to_catalog(inference_engine, f"engines.{inference_engine.label}", overwrite=True)
