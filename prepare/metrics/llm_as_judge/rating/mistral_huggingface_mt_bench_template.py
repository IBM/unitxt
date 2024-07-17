from unitxt import add_to_catalog
from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.llm_as_judge import LLMAsJudge

model_list = ["mistralai/Mistral-7B-Instruct-v0.2"]
format = "formats.models.mistral.instruction"
template = "templates.response_assessment.rating.mt_bench_single_turn"
task = "rating.single_turn"

for model_id in model_list:
    inference_model = HFPipelineBasedInferenceEngine(
        model_name=model_id, max_new_tokens=256, use_fp16=True
    )
    model_label = model_id.split("/")[1].replace("-", "_").replace(".", "_").lower()
    model_label = f"{model_label}_huggingface"
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
