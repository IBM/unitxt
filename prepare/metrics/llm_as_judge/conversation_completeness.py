from unitxt import add_to_catalog
from unitxt.inference import (
    GenericInferenceEngine,
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams,
)
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

template_name = "templates.response_assessment.judges.completeness.v5"

# inference_models = {
#     "llama3_1_v1_ibmgenai": {
#         "model_name": "llama3-1-70binstruct",
#         "inference_model": IbmGenAiInferenceEngine(
#             model_name="meta-llama/llama-3-1-70b-instruct",
#             parameters=IbmGenAiInferenceEngineParams(max_new_tokens=256),
#         ),
#     },
#     "generic_inference_engine": {
#         "model_name": "generic",
#         "inference_model": (GenericInferenceEngine()),
#     },
# }

inference_models = {
    "llama3_1_v1_ibmgenai": {
         "model_name": "llama3-1-70b-instruct",
         "inference_model": CrossProviderInferenceEngine(
            model="llama-3-1-70b-instruct", provider="rits"
        )
    }
}

for label, inference_model in inference_models.items():
    model_label = inference_model["model_name"]
    template_label = template_name.split(".")[-1]
    metric_label = (
        "metrics.llm_as_judge.rating." + model_label + "_template_" + template_label
    )

    cur_metric = LLMAsJudge(
        inference_model=inference_model["inference_model"],
        template=template_name,
        task="rating.single_turn",
        main_score=metric_label,
        prediction_type="str",
    )
    
    add_to_catalog(
        cur_metric,
        f"metrics.llm_as_judge.conversation_answer_completeness.{label}_judges",
        overwrite=True,
    )
