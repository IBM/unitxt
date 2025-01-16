from typing import Union

from unitxt import add_to_catalog, get_logger
from unitxt.llm_as_judge import LLMJudgeDirect, LLMJudgePairwise
from unitxt.llm_as_judge_constants import (
    DIRECT_CRITERIAS,
    EVALUATOR_TO_MODEL_ID,
    EVALUATORS_METADATA,
    INFERENCE_ENGINE_NAME_TO_CLASS,
    PAIRWISE_CRITERIAS,
    EvaluatorNameEnum,
    EvaluatorTypeEnum,
    ModelProviderEnum,
)
from unitxt.llm_as_judge_utils import get_evaluator_metadata, rename_model_if_required

logger = get_logger()


def get_evaluator(
    name: EvaluatorNameEnum,
    evaluator_type: EvaluatorTypeEnum,
    provider: ModelProviderEnum,
) -> Union[LLMJudgeDirect, LLMJudgePairwise]:
    evaluator_metadata = get_evaluator_metadata(name)

    inference_params = {"max_tokens": 1024, "seed": 42}
    model_name = rename_model_if_required(EVALUATOR_TO_MODEL_ID[name], provider)
    if provider == ModelProviderEnum.WATSONX:
        model_name = f"watsonx/{model_name}"
    elif provider == ModelProviderEnum.OPENAI:
        model_name = f"openai/{model_name}"

    params = {
        f"{'model' if provider != ModelProviderEnum.RITS else 'model_name'}": model_name,
        **inference_params,
    }

    inference_engine = INFERENCE_ENGINE_NAME_TO_CLASS[provider](**params)

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


logger.debug("Registering criterias...")
# Register all the predefined criterisa
for criteria in DIRECT_CRITERIAS:
    add_to_catalog(
        criteria,
        f"metrics.llm_as_judge.direct.criterias.{criteria.name}",
        overwrite=True,
    )

for criteria in PAIRWISE_CRITERIAS:
    add_to_catalog(
        criteria,
        f"metrics.llm_as_judge.pairwise.criterias.{criteria.name}",
        overwrite=True,
    )

logger.debug("Registering evaluators...")
for evaluator_metadata in EVALUATORS_METADATA:
    if evaluator_metadata.name not in [
        EvaluatorNameEnum.GRANITE_GUARDIAN_2B,
        EvaluatorNameEnum.GRANITE_GUARDIAN_8B,
    ]:
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
