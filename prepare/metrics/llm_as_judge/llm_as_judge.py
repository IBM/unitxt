from typing import Union

from unitxt import add_to_catalog, get_logger
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.llm_as_judge import LLMJudgeDirect, LLMJudgePairwise
from unitxt.llm_as_judge_constants import (
    DIRECT_CRITERIA,
    EVALUATOR_TO_MODEL_ID,
    EVALUATORS_METADATA,
    PAIRWISE_CRITERIA,
    EvaluatorNameEnum,
    EvaluatorTypeEnum,
    ModelProviderEnum,
)
from unitxt.llm_as_judge_utils import get_evaluator_metadata

logger = get_logger()


def get_evaluator(
    name: EvaluatorNameEnum,
    evaluator_type: EvaluatorTypeEnum,
    provider: ModelProviderEnum,
) -> Union[LLMJudgeDirect, LLMJudgePairwise]:
    evaluator_metadata = get_evaluator_metadata(name)
    inference_params = {"max_tokens": 1024, "seed": 42}
    model_name = EVALUATOR_TO_MODEL_ID[name]

    if provider == ModelProviderEnum.AZURE_OPENAI:
        inference_params["credentials"] = {}
        inference_params["credentials"]["api_base"] = (
            f"https://eteopenai.azure-api.net/openai/deployments/{model_name}/chat/completions?api-version=2024-08-01-preview"
        )

    inference_params["model"] = model_name

    inference_engine = CrossProviderInferenceEngine(**inference_params)

    params = {
        "inference_engine": inference_engine,
        "evaluator_name": evaluator_metadata.name.name,
        "generate_summaries": False,
    }

    evaluator_klass = (
        LLMJudgeDirect
        if evaluator_type == EvaluatorTypeEnum.DIRECT
        else LLMJudgePairwise
    )

    return evaluator_klass(**params)


logger.debug("Registering criteria...")
# Register all the predefined criterisa
for criteria in DIRECT_CRITERIA:
    add_to_catalog(
        criteria,
        f"metrics.llm_as_judge.direct.criteria.{criteria.name}",
        overwrite=True,
    )

for criteria in PAIRWISE_CRITERIA:
    add_to_catalog(
        criteria,
        f"metrics.llm_as_judge.pairwise.criteria.{criteria.name}",
        overwrite=True,
    )

logger.debug("Registering evaluators...")
for evaluator_metadata in EVALUATORS_METADATA:
    for provider in evaluator_metadata.providers:
        for evaluator_type in [
            EvaluatorTypeEnum.DIRECT,
            EvaluatorTypeEnum.PAIRWISE,
        ]:
            evaluator = get_evaluator(
                name=evaluator_metadata.name,
                evaluator_type=evaluator_type,
                provider=provider,
            )

            metric_name = (
                evaluator_metadata.name.value.lower()
                .replace("-", "_")
                .replace(".", "_")
                .replace(" ", "_")
            )
            add_to_catalog(
                evaluator,
                f"metrics.llm_as_judge.{evaluator_type.value}.{provider.value.lower()}.{metric_name}",
                overwrite=True,
            )
