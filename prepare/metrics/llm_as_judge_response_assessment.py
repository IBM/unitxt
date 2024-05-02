from unitxt import add_to_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge

"""
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
"""

model_list = ["meta-llama/llama-3-8b-instruct", "meta-llama/llama-3-70b-instruct"]

for model_name in model_list:
    inference_model = IbmGenAiInferenceEngine(model_name=model_name, max_new_tokens=252)
    recipe = (
        "card=cards.rag.model_response_assessment.llm_as_judge_using_mt_bench_template,"
        "template=templates.rag.model_response_assessment.llm_as_judge_using_mt_bench_template,"
        "demos_pool_size=0,"
        "num_demos=0"
    )
    model_label = model_name.split("/")[1].replace("-", "_")
    main_score = f"llm_as_judge_by_{model_label}_on_ibm_genai_using_mt_bench_template"
    metric = LLMAsJudge(
        inference_model=inference_model, recipe=recipe, main_score=main_score
    )

    add_to_catalog(
        metric,
        f"metrics.rag.model_response_assessment.llm_as_judge_by_{model_label}_on_ibm_genai_using_mt_bench_template",
        overwrite=True,
    )
