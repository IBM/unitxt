from unitxt import add_to_catalog
from unitxt.inference import HFPipelineBasedInferenceEngine, OpenAiInferenceEngine, OpenAiInferenceEngineParams
from unitxt.llm_as_judge import LLMAsJudge

model_list = ["gpt-4o"]
template = "templates.response_assessment.rating.mt_bench_single_turn"
task = "rating.single_turn"

for model_id in model_list:
    params = OpenAiInferenceEngineParams(max_tokens=1024)
    inference_model = OpenAiInferenceEngine(model_name=model_id, parameters=params)
    model_label = model_id.replace("-", "_").replace(".", "_").lower()
    model_label = f"{model_label}_openai"
    template_label = template.split(".")[-1]
    metric_label = f"{model_label}_template_{template_label}"
    metric = LLMAsJudge(
        inference_model=inference_model,
        template=template,
        task=task,
        main_score=metric_label,
    )

    add_to_catalog(
        metric,
        f"metrics.llm_as_judge.rating.{model_label}_template_{template_label}",
        overwrite=True,
    )
