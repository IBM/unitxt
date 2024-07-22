from unitxt import add_to_catalog
from unitxt.inference import IbmGenAiInferenceEngine
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.random_utils import get_seed

model_list = ["meta-llama/llama-3-8b-instruct", "meta-llama/llama-3-70b-instruct"]
format = "formats.llama3_instruct"
template = "templates.response_assessment.rating.mt_bench_single_turn"
task = "rating.single_turn"

for model_id in model_list:
    inference_model = IbmGenAiInferenceEngine(
        model_name=model_id, max_new_tokens=252, random_seed=get_seed()
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
