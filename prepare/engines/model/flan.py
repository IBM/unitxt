from unitxt.catalog import add_to_catalog
from unitxt.inference_engines import HFPipelineBasedInferenceEngine

engine = HFPipelineBasedInferenceEngine(
    model_name="google/flan-t5-small", max_new_tokens=32
)

add_to_catalog(engine, "engines.model.flan.t5_small.hf", overwrite=True)
