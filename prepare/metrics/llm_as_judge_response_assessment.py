from unitxt import add_to_catalog
from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.llm_as_judge import LLMAsJudge

inference_model = HFPipelineBasedInferenceEngine(
    model_name="google/flan-t5-large", max_new_tokens=32
)
recipe = (
    "card=cards.rag.model_response_assessment.llm_as_judge_using_mt_bench_template,"
    "template=templates.rag.model_response_assessment.llm_as_judge_using_mt_bench_template,"
    "demos_pool_size=0,"
    "num_demos=0"
)
main_score = "llm_as_judge_by_flan_t5_large_on_hf_pipeline_using_mt_bench_template"
metric = LLMAsJudge(
    inference_model=inference_model, recipe=recipe, main_score=main_score
)

add_to_catalog(
    metric,
    "metrics.rag.model_response_assessment.llm_as_judge_by_flan_t5_large_on_hf_pipeline_using_mt_bench_template",
    overwrite=True,
)
