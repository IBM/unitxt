from unitxt import add_to_catalog
from unitxt.evalassist_judge import EvalAssistLLMJudgeDirect
from unitxt.inference import CrossProviderInferenceEngine

for provider in ["watsonx", "rits"]:
    for model in ["llama-3-3-70b-instruct"]:
        eval_assist_judge = EvalAssistLLMJudgeDirect(
            inference_engine=CrossProviderInferenceEngine(
                provider=provider,
                model=model,
                max_tokens=1024,
                temperature=0.0,
            )
        )
        if model == "llama-3-3-70b-instruct":
            catalog_model = "llama3_3_70b"
        else:
            raise ValueError(f"Model {model} not supported")

        add_to_catalog(
            eval_assist_judge,
            f"metrics.llm_as_judge.evalassist.direct.{provider}.{catalog_model}",
            overwrite=True,
        )
