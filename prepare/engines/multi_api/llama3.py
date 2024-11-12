from unitxt.catalog import add_to_catalog
from unitxt.inference import MultiAPIInferenceEngine

engine = MultiAPIInferenceEngine(
    model="llama-3-8b-instruct",
    api_model_map={
        "watsonx": {
            "llama-3-8b-instruct": "watsonx/meta-llama/llama-3-8b-instruct",
        },
        "together-ai": {
            "llama-3-8b-instruct": "together_ai/togethercomputer/llama-3-8b-instruct"
        },
    },
)

add_to_catalog(engine, "engines.model.llama_3_8b_instruct", overwrite=True)
