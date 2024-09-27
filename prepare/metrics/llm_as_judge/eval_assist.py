
import os
from unitxt import add_to_catalog
from unitxt.inference import IbmGenAiInferenceEngine, IbmGenAiInferenceEngineParamsMixin
from unitxt.evalassist_llm_as_judge import (
    EvalAssistLLMAsJudge
)
from templates_eval_assist import template_dict
from rubrics_eval_assist import rubrics

# Register all the predefined rubrics
for rubric_name, rubric_obj in rubrics.items():
    add_to_catalog(
        rubric_obj,
        f"metrics.llm_as_judge.eval_assist.direct.rubrics.{rubric_name}",
        overwrite=True
    )

# os.environ["GENAI_KEY"] = ""
params = IbmGenAiInferenceEngineParamsMixin(max_new_tokens=1024, random_seed=42)

# Register the metrics for all four models
for model_name, inference_engine in [("mixtral", IbmGenAiInferenceEngine(model_name="mistralai/mixtral-8x7b-instruct-v01", parameters=params)),
                    ("granite", IbmGenAiInferenceEngine(model_name="ibm/granite-20b-code-instruct", parameters=params)),
                    ("llama_8b", IbmGenAiInferenceEngine(model_name="meta-llama/llama-3-8b-instruct", parameters=params)),
                    ("llama_70b", IbmGenAiInferenceEngine(model_name="meta-llama/llama-3-70b-instruct", parameters=params)),
                    ("prometheus", IbmGenAiInferenceEngine(model_name="kaist-ai/prometheus-8x7b-v2", parameters=params))]:
    eval_assist_metric = EvalAssistLLMAsJudge(inference_model=inference_engine, 
                                            #   rubric=rubrics["temperature"],
                                              assessment_template=template_dict[model_name]["assessment"],
                                              summ_template=template_dict[model_name]["summarization"],
                                              answer_template=template_dict[model_name]["answer"])
    
    add_to_catalog(
        eval_assist_metric,
        f"metrics.llm_as_judge.eval_assist.direct.{model_name}",
        overwrite=True
    )
