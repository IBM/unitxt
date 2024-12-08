from unitxt.catalog import add_to_catalog
from unitxt.inference import CrossProviderInferenceEngine

model_list = ["meta-llama/llama-3-8b-instruct", "meta-llama/llama-3-70b-instruct"]

for model in model_list:
    model_label = model.split("/")[1].replace("-", "_").replace(".", ",").lower()
    inference_model = CrossProviderInferenceEngine(
        model=model, provider="watsonx", max_tokens=2048, seed=42
    )
    add_to_catalog(
        inference_model, f"engines.cross_provider.{model_label}", overwrite=True
    )
