from unitxt import add_to_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams,
    GenericInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge

template_name = "templates.response_assessment.judges.idk.v1"

inference_models = {
    "llama3_v1_ibmgenai" : {
        "model_name": "llama370binstruct",
        "inference_model": IbmGenAiInferenceEngine(model_name="meta-llama/llama-3-70b-instruct", 
    parameters=IbmGenAiInferenceEngineParams(max_new_tokens=256))},
    "generic_inference_engine": {
        "model_name" :"generic",
        "inference_model" : (GenericInferenceEngine())
    }
}

for label,inference_model in inference_models.items():
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

# _description__=  "Does the model response say I don't know?"

    add_to_catalog(
        cur_metric,
        f"metrics.llm_as_judge.conversation_answer_idk.{label}_judges",
        overwrite=True,
    )
