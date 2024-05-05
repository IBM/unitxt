from unitxt import add_to_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams,
)
from unitxt.llm_as_judge import LLMAsJudge

model_list = ["meta-llama/llama-3-8b-instruct", "meta-llama/llama-3-70b-instruct"]

gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=252)

for model_name in model_list:
    inference_model = IbmGenAiInferenceEngine(
        model_name=model_name, parameters=gen_params
    )
    recipe = (
        "card=cards.rag.model_response_assessment.llm_as_judge_metric_absolute_score_single_turn,"
        "template=templates.model_response_assessment.mt_bench_absolute_score_single_turn,"
        "demos_pool_size=0,"
        "num_demos=0"
    )
    # llm_as_judge_metric_absolute_score_single_turn_mt_bench_template
    model_label = model_name.split("/")[1].replace("-", "_")
    main_score = f"llm_as_judge_metric_absolute_score_single_turn_mt_bench_template_using_{model_label}_on_ibm_genai"
    metric = LLMAsJudge(
        inference_model=inference_model, recipe=recipe, main_score=main_score
    )

    add_to_catalog(
        metric,
        f"metrics.rag.model_response_assessment.llm_as_judge_metric_absolute_score_single_turn_mt_bench_template_using_{model_label}_on_ibm_genai",
        overwrite=True,
    )
