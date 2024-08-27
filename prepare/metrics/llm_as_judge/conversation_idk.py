from unitxt import add_to_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams,
)
from unitxt.llm_as_judge import LLMAsJudge

platform = "ibm_gen_ai"
gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=256)

model_name = "meta-llama/llama-3-70b-instruct"
template_name = "templates.response_assessment.judges.idk.v1"

inference_model = IbmGenAiInferenceEngine(model_name=model_name, parameters=gen_params)

model_label = model_name.split("/")[1].replace("-", "")
template_label = template_name.split(".")[-1]

metric_label = (
    "metrics.llm_as_judge.rating." + model_label + "_template_" + template_label
)

cur_metric = LLMAsJudge(
    inference_model=inference_model,
    template=template_name,
    task="rating.single_turn",
    main_score=metric_label,
    prediction_type="str",
)

# _description__=  "Does the model response say I don't know?"

add_to_catalog(
    cur_metric,
    "metrics.llm_as_judge.conversation_answer_idk.llama3_v1_ibmgenai_judges",
    overwrite=True,
)
