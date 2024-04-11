from unitxt import add_to_catalog
from unitxt.inference import PipelineBasedInferenceEngine
from unitxt.llm_as_judge import LLMAsJudge


inference_model = PipelineBasedInferenceEngine(model_name='google/flan-t5-large', max_new_tokens=32)

metric = LLMAsJudge(inference_model=inference_model)

add_to_catalog(
    metric,
    "metrics.llm_as_judge.model_response_assessment.mt_bench_flan_t5",
    overwrite=True,
)
