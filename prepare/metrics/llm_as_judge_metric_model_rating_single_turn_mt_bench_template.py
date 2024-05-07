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
        "card": "cards.rag.model_response_assessment.model_rating_single_turn",
        "template": "templates.model_response_assessment.mt_bench_model_rating_single_turn",
    },
    {
        "card": "cards.rag.model_response_assessment.model_rating_with_reference_single_turn",
        "template": "templates.model_response_assessment.mt_bench_model_rating_with_reference_single_turn",
    },
    {
        "card": "cards.rag.model_response_assessment.model_rating_multi_turn",
        "template": "templates.model_response_assessment.mt_bench_model_rating_multi_turn",
    },
    {
        "card": "cards.rag.model_response_assessment.model_rating_with_reference_multi_turn",
        "template": "templates.model_response_assessment.mt_bench_model_rating_with_reference_multi_turn",
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
        card_label = card.split(".")[-1]
        metric_label = f"llm_as_judge_{card_label}_{model_label}_ibm_genai"
        metric = LLMAsJudge(
            inference_model=inference_model, recipe=recipe, main_score=metric_label
        )

        add_to_catalog(
            metric,
            f"metrics.rag.model_response_assessment.{metric_label}",
            overwrite=True,
        )
