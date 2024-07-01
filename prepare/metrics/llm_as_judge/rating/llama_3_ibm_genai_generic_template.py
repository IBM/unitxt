from unitxt import add_to_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams,
)
from unitxt.llm_as_judge import LLMAsJudge

model = "meta-llama/llama-3-70b-instruct"
format = "formats.llama3_chat"
template = "templates.response_assessment.rating.generic_single_turn"
task = "rating.single_turn"

gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=252)
inference_model = IbmGenAiInferenceEngine(model_name=model, parameters=gen_params)
model_label = model.split("/")[1].replace("-", "_").replace(".", ",").lower()
model_label = f"{model_label}_ibm_genai"
template_label = template.split(".")[-1]
metric_label = f"{model_label}_template_{template_label}"
metric = LLMAsJudge(
    inference_model=inference_model,
    template=template,
    task=task,
    format=format,
    main_score=metric_label,
)

add_to_catalog(
    metric,
    f"metrics.llm_as_judge.rating.{model_label}_template_{template_label}",
    overwrite=True,
)
