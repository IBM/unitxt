from unitxt import add_to_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge

model_list = [
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3-70b-instruct",
]  # will point to llamaguard2
format = "formats.llama3_chat"
template = "templates.safety.unsafe_content"
task = "rating.single_turn"

gen_params = {"max_new_tokens": 252}
for model_id in model_list:
    inference_model = IbmGenAiInferenceEngine(
        model_name=model_id, parameters=gen_params
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
    )

    add_to_catalog(
        metric,
        f"metrics.llm_as_judge.safety.{model_label}_template_{template_label}",
        overwrite=True,
    )
