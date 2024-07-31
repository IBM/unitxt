from unitxt import add_to_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams,
)
from unitxt.llm_as_judge import LLMAsJudge

model_list = ["meta-llama/llama-3-70b-instruct"]
format = "formats.llama3_instruct"
template = "templates.response_assessment.rating.table2text_single_turn_with_reference"
task = "rating.single_turn_with_reference"

gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=252)
for model_id in model_list:
    inference_model = IbmGenAiInferenceEngine(
        model_name=model_id,
        parameters=gen_params,
        data_classification_policy=["public"],
    )
    model_label = model_id.split("/")[1].replace("-", "_").replace(".", ",").lower()
    model_label = f"{model_label}_ibm_genai"
    template_label = template.split(".")[-1]
    metric_label = f"{model_label}_template_{template_label}"
    metric = LLMAsJudge(
        inference_model=inference_model,
        template=template,
        task=task,
        format=format,
        main_score=metric_label,
        prediction_type="str",
    )

    add_to_catalog(
        metric,
        f"metrics.llm_as_judge.rating.{model_label}_template_{template_label}",
        overwrite=True,
    )
