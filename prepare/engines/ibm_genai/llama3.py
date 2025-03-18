from unitxt.catalog import add_to_catalog
from unitxt.inference import IbmGenAiInferenceEngine

model_list = ["meta-llama/llama-3-8b-instruct", "meta-llama/llama-3-70b-instruct"]

for model in model_list:
    model_label = model.split("/")[1].replace("-", "_").replace(".", ",").lower()
    inference_model = IbmGenAiInferenceEngine(
        model_name=model, max_new_tokens=2048, random_seed=42
    )
    add_to_catalog(inference_model, f"engines.ibm_gen_ai.{model_label}", overwrite=True)
