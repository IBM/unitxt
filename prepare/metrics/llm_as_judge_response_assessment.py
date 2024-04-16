from unitxt import add_to_catalog
from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.llm_as_judge import LLMAsJudge

inference_model = HFPipelineBasedInferenceEngine(
    model_name="google/flan-t5-large", max_new_tokens=32
)
recipe = (
    "card=cards.llm_as_judge.model_response_assessment.mt_bench,"
    "template=templates.llm_as_judge.model_response_assessment.mt_bench,"
    "demos_pool_size=0,"
    "num_demos=0"
)

metric = LLMAsJudge(inference_model=inference_model, recipe=recipe)

add_to_catalog(
    metric,
    "metrics.rag.model_response_assessment.llm_as_judge_by_flan_t5_large_on_hf_pipeline_using_mt_bench_template",
    overwrite=True,
)
