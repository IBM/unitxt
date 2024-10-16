import os
from unitxt import add_to_catalog
from unitxt.inference import IbmGenAiInferenceEngineParamsMixin, IbmGenAiInferenceEngine, OpenAiInferenceEngine, OpenAiInferenceEngineParamsMixin
from unitxt.evalassist_llm_as_judge_direct import (
    EvalAssistLLMAsJudgeDirect
)
from unitxt.eval_assist_constants import AvailableDirectEvaluators

from templates_eval_assist import template_dict
from rubrics_eval_assist import rubrics

print("registering rubrics")
# Register all the predefined rubrics
for rubric_name, rubric_obj in rubrics.items():
    print(rubric_name)
    add_to_catalog(
        rubric_obj,
        f"metrics.llm_as_judge.eval_assist.direct.rubrics.{rubric_name}",
        overwrite=True
    )

os.environ["GENAI_KEY"] = ""
params = IbmGenAiInferenceEngineParamsMixin(max_new_tokens=1024, random_seed=42)
os.environ["OPENAI_API_KEY"] = ""
openai_params = OpenAiInferenceEngineParamsMixin(max_tokens=1024, seed=42)

print("registering evaluators")
for evaluator in AvailableDirectEvaluators:
    print(evaluator.value)
    if "gpt" in evaluator.value :
        inference_engine = OpenAiInferenceEngine(model_name=evaluator.model_id, parameters=openai_params)
    else:
        inference_engine =  IbmGenAiInferenceEngine(model_name=evaluator.model_id, parameters=params)

    eval_assist_metric = EvalAssistLLMAsJudgeDirect(inference_model=inference_engine,
                                              assessment_template=template_dict[evaluator.json_name]["assessment"],
                                              summ_template=template_dict[evaluator.json_name]["summarization"],
                                              answer_template=template_dict[evaluator.json_name]["answer"])
    add_to_catalog(
        eval_assist_metric,
        f"metrics.llm_as_judge.eval_assist.direct.{evaluator.json_name}",
        overwrite=True
    )
