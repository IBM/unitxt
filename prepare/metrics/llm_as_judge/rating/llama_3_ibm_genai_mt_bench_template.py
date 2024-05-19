from unitxt import add_to_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams,
)
from unitxt.llm_as_judge import LLMAsJudgeSingleModelSingleTurn

model_list = ["meta-llama/llama-3-8b-instruct", "meta-llama/llama-3-70b-instruct"]
format = "formats.llama3_chat"
template = "templates.response_assessment.rating.mt_bench_single_turn"
template_model_input_field_name = "question"
template_model_output_model_field_name = "answer"
template_reference_field_name = None

gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=252)
for model_id in model_list:
    inference_model = IbmGenAiInferenceEngine(
        model_name=model_id, parameters=gen_params
    )
    model_label = model_id.split("/")[1].replace("-", "_").replace(".", ",").lower()
    model_label = f"{model_label}_ibm_genai"
    template_label = template.split(".")[-1]
    metric_label = f"{model_label}_template_{template_label}"
    metric = LLMAsJudgeSingleModelSingleTurn(
        inference_model=inference_model,
        template=template,
        template_model_input_field_name=template_model_input_field_name,
        template_model_output_field_name=template_model_output_model_field_name,
        template_reference_field_name=template_reference_field_name,
        format=format,
        main_score=metric_label,
    )

    add_to_catalog(
        metric,
        f"metrics.llm_as_judge.rating.{model_label}_template_{template_label}",
        overwrite=True,
    )
