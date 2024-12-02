from unitxt.catalog import add_to_catalog
from unitxt.inference import OpenAiInferenceEngine

model_name = "gpt-4o"
model_label = model_name.replace("-", "_").lower()
inference_model = OpenAiInferenceEngine(model_name=model_name, max_tokens=2048, seed=42)
add_to_catalog(inference_model, f"engines.openai.{model_label}", overwrite=True)