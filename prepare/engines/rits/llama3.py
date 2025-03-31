from unitxt.catalog import add_to_catalog
from unitxt.inference import RITSInferenceEngine

model_list = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/llama-3-1-70b-instruct",
    "meta-llama/llama-3-1-405b-instruct-fp8",
]

for model in model_list:
    model_label = model.split("/")[1].replace("-", "_").replace(",", "_").lower()
    inference_model = RITSInferenceEngine(model_name=model, max_tokens=2048, seed=42)
    add_to_catalog(inference_model, f"engines.rits.{model_label}", overwrite=True)
