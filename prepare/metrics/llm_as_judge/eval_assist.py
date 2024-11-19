from typing import Optional, Union
from unitxt import add_to_catalog
from unitxt.eval_assist_llm_as_judge_direct import EvalAssistLLMAsJudgeDirect
from unitxt.eval_assist_llm_as_judge_pairwise import EvalAssistLLMAsJudgePairwise
from unitxt.eval_assist_constants import DIRECT_ASSESSMENT_CRITERIAS, EVALUATOR_TO_MODEL_ID, EVALUATORS_METADATA, PAIRWISE_COMPARISON_CRITERIAS, EvaluatorMetadata, EvaluatorNameEnum, EvaluatorTypeEnum, INFERENCE_ENGINE_NAME_TO_CLASS
from templates_eval_assist_direct import direct_assessment_template_dict
from templates_eval_assist_pairwise import pairwise_comparison_template_dict



def get_evaluator_metadata(name: EvaluatorNameEnum) -> EvaluatorMetadata: #, evaluator_type: EvaluatorTypeEnum) -> EvaluatorMetadata:
    evaluator_search = [e for e in EVALUATORS_METADATA if e.name == name] #and e.evaluator_type == evaluator_type]
    if len(evaluator_search) == 0:
        # raise ValueError(f'A {evaluator_type} evaluator with id {name} does not exist.')
        raise ValueError(f'An evaluator with id {name} does not exist.')
    if len(evaluator_search) > 1:
        # raise ValueError(f'A {evaluator_type} evaluator with id {name} matched several models.')
        raise ValueError(f'An evaluator with id {name} matched several models.')
    return evaluator_search[0]

def get_evaluator(
        name: EvaluatorNameEnum,
        evaluator_type: EvaluatorTypeEnum,
        credentials: Optional[dict[str,str]] = None,
        provider: Optional[str] = None,
        inference_engine_params: Optional[dict[str,any]] = None) ->  EvalAssistLLMAsJudgeDirect | EvalAssistLLMAsJudgePairwise:
    
    evaluator_metadata = get_evaluator_metadata(name) #, evaluator_type)

    selected_provider = provider
    if selected_provider is None:
        if len(evaluator_metadata.providers) > 1:
            raise Exception(f'Evaluator {evaluator_metadata.name} has more than one available model providers ({evaluator_metadata.providers}). Use the provider parameter to select one of them.')
        selected_provider = evaluator_metadata.providers[0]

    params = {"model_name": EVALUATOR_TO_MODEL_ID[evaluator_metadata.name]}
    params = {
        **params,
        **inference_engine_params
    }
    inference_engine = INFERENCE_ENGINE_NAME_TO_CLASS[selected_provider](**params)

    model_family = evaluator_metadata.model_family
    if evaluator_type == EvaluatorTypeEnum.DIRECT_ASSESSMENT:
        evaluator = EvalAssistLLMAsJudgeDirect(
            inference_engine=inference_engine,
            assessment_template=direct_assessment_template_dict[model_family]["assessment"],
            summarization_template=direct_assessment_template_dict[model_family]["summarization"],
            option_selection_template=direct_assessment_template_dict[model_family]["answer"],
            option_selection_strategy=evaluator_metadata.option_selection_strategy.name,
            evaluator_name = evaluator_metadata.name.name,
            model_family=evaluator_metadata.model_family.name
        )
    else:
        evaluator = EvalAssistLLMAsJudgePairwise(
            inference_engine=inference_engine,
            assessment_template=pairwise_comparison_template_dict[model_family]["assessment"],
            summarization_template=pairwise_comparison_template_dict[model_family]["summarization"],
            option_selection_template=pairwise_comparison_template_dict[model_family]["answer"],
            option_selection_strategy=evaluator_metadata.option_selection_strategy.name,
            evaluator_name = evaluator_metadata.name.name
        )
    return evaluator

print("Registering criterias...")
# Register all the predefined criterisa
for criteria in DIRECT_ASSESSMENT_CRITERIAS:
    add_to_catalog(
        criteria,
        f"metrics.llm_as_judge.eval_assist.direct_assessment.criterias.{criteria.name}",
        overwrite=True
    )

for criteria in PAIRWISE_COMPARISON_CRITERIAS:
    add_to_catalog(
        criteria,
        f"metrics.llm_as_judge.eval_assist.pairwise_comparison.criterias.{criteria.name}",
        overwrite=True
    )
    
print("Registering evaluators...")
for evaluator_metadata in EVALUATORS_METADATA:
    for provider in evaluator_metadata.providers:
        for evaluator_type in [EvaluatorTypeEnum.DIRECT_ASSESSMENT, EvaluatorTypeEnum.PAIRWISE_COMPARISON]:
            evalutor = get_evaluator(
                name=evaluator_metadata.name,
                evaluator_type=evaluator_type,
                provider=provider,
                inference_engine_params={
                    "max_new_tokens": 1024,
                    "random_seed": 42
                } if evaluator_metadata.name != EvaluatorNameEnum.GPT4 else {
                    "max_tokens": 1024,
                    "seed": 42
                }
            )

            metric_name = evaluator_metadata.name.value.lower().replace('-', '_').replace('.','_')
            add_to_catalog(
                evalutor,
                f"metrics.llm_as_judge.eval_assist.{evaluator_type.value}.{metric_name}",
                overwrite=True
            )
        