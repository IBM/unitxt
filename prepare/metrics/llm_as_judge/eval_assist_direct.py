from unitxt import add_to_catalog
from unitxt.inference import IbmGenAiInferenceEngine, OpenAiInferenceEngine
from unitxt.evalassist_llm_as_judge_direct import (
    EvalAssistLLMAsJudgeDirect
)
from unitxt.eval_assist_constants import AvailableDirectEvaluators, AvailableRubrics
from templates_eval_assist import template_dict

print("registering rubrics")
# Register all the predefined rubrics

for available_rubric in AvailableRubrics:
    add_to_catalog(
        available_rubric.rubric,
        f"metrics.llm_as_judge.eval_assist.direct.rubrics.{available_rubric.json_name}",
        overwrite=True
    )

print("registering evaluators")
for evaluator in AvailableDirectEvaluators:
    print(evaluator.json_name)
    if "gpt" in evaluator.json_name :
        inference_engine = OpenAiInferenceEngine(model_name=evaluator.model_id, data_classification_policy=["public"], max_tokens=1024, seed=4)
    else:
        inference_engine =  IbmGenAiInferenceEngine(model_name=evaluator.model_id, data_classification_policy=["public"], max_new_tokens=1024, random_seed=42)

    eval_assist_metric = EvalAssistLLMAsJudgeDirect(inference_model=inference_engine,
                                              assessment_template=template_dict[evaluator.json_name]["assessment"],
                                              summ_template=template_dict[evaluator.json_name]["summarization"],
                                              answer_template=template_dict[evaluator.json_name]["answer"])
    add_to_catalog(
        eval_assist_metric,
        f"metrics.llm_as_judge.eval_assist.direct.{evaluator.json_name}",
        overwrite=True
    )
    