
import os
from unitxt import add_to_catalog
from unitxt.inference import IbmGenAiInferenceEngine, IbmGenAiInferenceEngineParamsMixin
from unitxt.evalassist_llm_as_judge_pairwise import (
    EvalAssistLLMAsJudgePairwise
)
from unitxt.eval_assist_constants import AvailablePairwiseEvaluators, AvailablePairwiseCriterias

from templates_eval_assist_pairwise import pairwise_template_dict


# Register pre-defined criterias
for available_pairwise in AvailablePairwiseCriterias:
    add_to_catalog(
        available_pairwise.pairwise_criteria,
        f"metrics.llm_as_judge.eval_assist.direct.rubrics.{available_pairwise.json_name}",
        overwrite=True
    )

params = IbmGenAiInferenceEngineParamsMixin(max_new_tokens=1024, random_seed=42)
for evaluator in AvailablePairwiseEvaluators:
    inference_engine = IbmGenAiInferenceEngine(model_name=evaluator.model_id, parameters=params)
    eval_assist_metric = EvalAssistLLMAsJudgePairwise(inference_model=inference_engine, 
                                              assessment_template=pairwise_template_dict[evaluator.json_name]["assessment"],
                                              summ_template=pairwise_template_dict[evaluator.json_name]["summarization"],
                                              answer_template=pairwise_template_dict[evaluator.json_name]["answer"])
    
    add_to_catalog(
        eval_assist_metric,
        f"metrics.llm_as_judge.eval_assist.pairwise.{evaluator.json_name}",
        overwrite=True
    )
    