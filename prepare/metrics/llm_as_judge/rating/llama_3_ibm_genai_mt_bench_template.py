from unitxt import add_to_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams,
)
from unitxt.llm_as_judge import LLMAsJudge

model_and_format_list = [
    {"model_id": "meta-llama/llama-3-8b-instruct", "format": "formats.llama3_chat"},
    {"model_id": "meta-llama/llama-3-70b-instruct", "format": "formats.llama3_chat"},
]

card_and_template_list = [
    {
        "card": "cards.dynamic_cards_for_llm_judges.rating.single_turn",
        "template": "templates.response_assessment.rating.mt_bench_single_turn",
    },
    {
        "card": "cards.dynamic_cards_for_llm_judges.rating.single_turn_with_reference",
        "template": "templates.response_assessment.rating.mt_bench_single_turn_with_reference",
    },
]

gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=252)

for model_and_format_dict in model_and_format_list:
    model_id = model_and_format_dict["model_id"]
    model_format = model_and_format_dict["format"]
    inference_model = IbmGenAiInferenceEngine(
        model_name=model_id, parameters=gen_params
    )

    for card_and_template_dict in card_and_template_list:
        card = card_and_template_dict["card"]
        template = card_and_template_dict["template"]
        recipe = (
            f"card={card},"
            f"template={template},"
            f"format={model_format},"
            "demos_pool_size=0,"
            "num_demos=0"
        )
        model_label = model_id.split("/")[1].replace("-", "_")
        model_label = f"{model_label}_ibm_genai"
        template_label = template.split(".")[-1]
        metric_label = f"{model_label}_template_{template_label}"
        metric = LLMAsJudge(
            inference_model=inference_model, recipe=recipe, main_score=metric_label
        )

        card_subtask = card.split(".")[-1:]
        add_to_catalog(
            metric,
            f"metrics.llm_as_judge.rating.{model_label}_template_{template_label}",
            overwrite=True,
        )
