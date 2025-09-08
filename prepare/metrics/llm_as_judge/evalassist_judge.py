from unitxt import add_to_catalog
from unitxt.evalassist_judge import EvalAssistLLMJudgeDirect
from unitxt.inference import CrossProviderInferenceEngine

eval_assist_judge = EvalAssistLLMJudgeDirect(
    inference_engine=CrossProviderInferenceEngine(
        model="llama-3-3-70b-instruct",
        max_tokens=1024,
        temperature=0.0,
    ),
)

add_to_catalog(
    eval_assist_judge,
    "metrics.evalassist_judge.direct.watsonx.llama3_3_70b",
    overwrite=True,
)
