from unitxt import add_to_catalog
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.random_utils import get_seed

model_list = ["llama-3-1-70b-instruct"]
template = "templates.response_assessment.rating.table2text_single_turn_with_reference"

for model_id in model_list:
    inference_model = CrossProviderInferenceEngine(
        model=model_id, max_tokens=252, seed=get_seed()
    )
    model_label = (
        model_id.replace("-", "_").replace(".", ",").lower() + "_cross_provider"
    )
    template_label = template.split(".")[-1]
    metric = LLMAsJudge(
        inference_model=inference_model,
        template=template,
        task="rating.single_turn_with_reference",
        format="formats.chat_api",
        main_score=f"{model_label}_template_{template_label}",
        prediction_type="str",
    )
    add_to_catalog(
        metric,
        f"metrics.llm_as_judge.rating.{model_label}_template_{template_label}",
        overwrite=True,
    )
