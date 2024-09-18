import json

from unitxt import add_to_catalog
from unitxt.inference import IbmGenAiInferenceEngine, IbmGenAiInferenceEngineParamsMixin
from unitxt.evalassist_llm_as_judge import (
    EvalAssistLLMAsJudge
)
import os

config_filepath = "prepare/metrics/llm_as_judge/eval_assist.json"

with open(config_filepath) as file:
    config = json.load(file)
print("config is ", config)

os.environ["GENAI_KEY"] = "pak-cAcH7ExLy-3jXXCICeCr-jQD3xK8grc3B32vczLXa9E"
params = IbmGenAiInferenceEngineParamsMixin(max_new_tokens=1024, random_seed=42)

for model_name, inference_engine in [("mixtral", IbmGenAiInferenceEngine(model_name="mistralai/mixtral-8x7b-instruct-v01", parameters=params)),
                    ("granite", IbmGenAiInferenceEngine(model_name="ibm/granite-20b-code-instruct", parameters=params)),
                    ("llama_8b", IbmGenAiInferenceEngine(model_name="meta-llama/llama-3-8b-instruct", parameters=params)),
                    ("llama_70b", IbmGenAiInferenceEngine(model_name="meta-llama/llama-3-70b-instruct", parameters=params))]:
    eval_assist_metric = EvalAssistLLMAsJudge(inference_model=inference_engine)
    add_to_catalog(
        eval_assist_metric,
        f"metrics.llm_as_judge.eval_assist.direct.{model_name}",
        overwrite=True
    )