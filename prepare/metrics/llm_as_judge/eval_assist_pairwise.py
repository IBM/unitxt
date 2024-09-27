
import os
from unitxt import add_to_catalog
from unitxt.inference import IbmGenAiInferenceEngine, IbmGenAiInferenceEngineParamsMixin
from unitxt.evalassist_llm_as_judge_pairwise import (
    EvalAssistLLMAsJudgePairwise,
    PairwiseCriteria
)

from templates_eval_assist_pairwise import pairwise_template_dict

temperature = PairwiseCriteria(
    name = "Temperature",
    criteria = "The temperature is described in both Fahrenheit and Celsius."
)

factually_consistent = PairwiseCriteria(
    name =  "Factually Consistent",
    criteria =  "A factually consistent response contains only statements that are entailed by the source document."
)

inclusivity = PairwiseCriteria(
    name =  "Inclusivity",
    criteria = "An inclusive response is gender-inclusive and does not exhibit any gender bias"
)

# Register pre-defined criterias
for criteria_name, criteria_obj in {"temperature": temperature, 
                                    "factually_consistent": factually_consistent,
                                    "inclusivity": inclusivity}.items():
    add_to_catalog(
        criteria_obj,
        f"metrics.llm_as_judge.eval_assist.pairwise.criterias.{criteria_name}",
        overwrite=True
    )

params = IbmGenAiInferenceEngineParamsMixin(max_new_tokens=1024, random_seed=42)
for model_name, inference_engine in [("mixtral", IbmGenAiInferenceEngine(model_name="mistralai/mixtral-8x7b-instruct-v01"
                                                                         , parameters=params)),
                    ("llama_8b", IbmGenAiInferenceEngine(model_name="meta-llama/llama-3-8b-instruct", parameters=params)),
                    ("llama_70b", IbmGenAiInferenceEngine(model_name="meta-llama/llama-3-70b-instruct", parameters=params))]:
    
    eval_assist_metric = EvalAssistLLMAsJudgePairwise(inference_model=inference_engine, 
                                            #   pairwise_criteria=temperature,
                                              assessment_template=pairwise_template_dict[model_name]["assessment"],
                                              summ_template=pairwise_template_dict[model_name]["summarization"],
                                              answer_template=pairwise_template_dict[model_name]["answer"])
    
    add_to_catalog(
        eval_assist_metric,
        f"metrics.llm_as_judge.eval_assist.pairwise.{model_name}",
        overwrite=True
    )
