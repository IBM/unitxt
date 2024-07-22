from unitxt import add_to_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge

model_list = ["meta-llama/llama-3-8b-instruct", "meta-llama/llama-3-70b-instruct"]
format = "formats.llama3_instruct"
templates = [
    "templates.response_assessment.pairwise_comparative_rating.arena_hard",
    "templates.response_assessment.pairwise_comparative_rating.arena_hard_with_shuffling",
]
for template in templates:
    task = "pairwise_comparative_rating.single_turn"

    for model_id in model_list:
        inference_model = IbmGenAiInferenceEngine(
            model_name=model_id, max_new_tokens=2048, random_seed=42
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
            f"metrics.llm_as_judge.pairwise_comparative_rating.{model_label}_template_{template_label}",
            overwrite=True,
        )
