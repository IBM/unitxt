from unitxt.catalog import add_to_catalog
from unitxt.inference import CrossProviderInferenceEngine

engine = CrossProviderInferenceEngine(
    model="llama-3-8b-instruct",
)

add_to_catalog(engine, "engines.model.llama_3_8b_instruct", overwrite=True)
