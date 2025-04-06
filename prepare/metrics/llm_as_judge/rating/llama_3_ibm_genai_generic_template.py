from unitxt import add_to_catalog
from unitxt.inference import CrossProviderInferenceEngine, IbmGenAiInferenceEngine
from unitxt.llm_as_judge_from_template import LLMAsJudge
from unitxt.random_utils import get_seed

inference_model = IbmGenAiInferenceEngine(
    model_name="meta-llama/llama-3-70b-instruct",
    max_new_tokens=252,
    random_seed=get_seed(),
)

metric = LLMAsJudge(
    inference_model=inference_model,
    template="templates.response_assessment.rating.generic_single_turn",
    task="rating.single_turn",
    format="formats.llama3_instruct",
    main_score="llama_3_70b_instruct_ibm_genai_template_generic_single_turn",
    prediction_type=str,
)

add_to_catalog(
    metric,
    "metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn",
    overwrite=True,
)

metric = LLMAsJudge(
    inference_model=inference_model,
    template="templates.response_assessment.rating.generic_single_turn_with_reference",
    task="rating.single_turn_with_reference",
    format="formats.llama3_instruct",
    single_reference_per_prediction=True,
    main_score="llama_3_70b_instruct_ibm_genai_template_generic_single_turn_with_reference",
)

add_to_catalog(
    metric,
    "metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn_with_reference",
    overwrite=True,
)


inference_model = CrossProviderInferenceEngine(
    model="llama-3-70b-instruct", max_tokens=252
)

metric = LLMAsJudge(
    inference_model=inference_model,
    template="templates.response_assessment.rating.generic_single_turn",
    task="rating.single_turn",
    format="formats.chat_api",
    main_score="llama_3_70b_instruct_template_generic_single_turn",
    prediction_type=str,
)

add_to_catalog(
    metric,
    "metrics.llm_as_judge.rating.llama_3_70b_instruct.generic_single_turn",
    overwrite=True,
)

metric = LLMAsJudge(
    inference_model=inference_model,
    template="templates.response_assessment.rating.generic_single_turn_with_reference",
    task="rating.single_turn_with_reference",
    format="formats.chat_api",
    single_reference_per_prediction=True,
    main_score="llama_3_70b_instruct_template_generic_single_turn_with_reference",
)

add_to_catalog(
    metric,
    "metrics.llm_as_judge.rating.llama_3_70b_instruct.generic_single_turn_with_reference",
    overwrite=True,
)
