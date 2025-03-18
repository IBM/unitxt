from unitxt.catalog import add_to_catalog
from unitxt.inference import OllamaInferenceEngine

inference_model = OllamaInferenceEngine(model="llama2")
add_to_catalog(inference_model, "engines.ollama.llama2", overwrite=True)
