from unitxt import add_to_catalog
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.llm_as_judge import LLMJudgeDirect
from unitxt.llm_as_judge_constants import (
    CriteriaWithOptions,
)

option_map = {
    "Excellent": 1.0,
    "Good": 0.75,
    "mediocre": 0.5,
    "Bad": 0.25,
    "Very Bad": 0,
}

# First, describe a judgement criteria
adherence_criteria = CriteriaWithOptions.from_obj(
    {
        "name": "adherence_with_format",
        "description": "The response aligns with the requested structure, style, or format (e.g., bullet points, headings, specific phrasing).",
        "options": [
            {
                "name": "Excellent",
                "description": "The response perfectly aligns with the requested structure, style, or format, with no deviations.",
            },
            {
                "name": "Good",
                "description": "The response aligns well with the requested structure, style, or format, with minor deviations that do not affect clarity or usability.",
            },
            {
                "name": "mediocre",
                "description": "The response generally follows the requested structure, style, or format, but noticeable inconsistencies or omissions are present.",
            },
            {
                "name": "Bad",
                "description": "The response only partially aligns with the requested structure, style, or format, with significant inconsistencies or a lack of adherence.",
            },
            {
                "name": "Very Bad",
                "description": "The response fails to align with the requested structure, style, or format.",
            },
        ],
        "option_map": option_map,
    }
)
add_to_catalog(
    adherence_criteria,
    f"metrics.llm_as_judge.direct.criterias.{adherence_criteria.name}",
    overwrite=True,
)

completeness_criteria = CriteriaWithOptions.from_obj(
    {
        "name": "answer_completeness",
        "description": "The response is complete: all the aspects of the reference answer are addressed in the response. The "
        "response might use different phrasing or wording from the reference answer.",
        "options": [
            {
                "name": "Excellent",
                "description": "The response addresses all aspects of the reference answer.",
            },
            {
                "name": "Good",
                "description": "The response addresses most aspects of the reference answer, with minor omissions.",
            },
            {
                "name": "mediocre",
                "description": "The response covers the essential aspects of the reference answer but has notable omissions.",
            },
            {
                "name": "Bad",
                "description": "The response covers only a few aspects of the reference answer, with significant omissions.",
            },
            {
                "name": "Very Bad",
                "description": "The response fails to address the reference answer meaningfully, with most aspects omitted.",
            },
        ],
        "option_map": option_map,
    }
)
add_to_catalog(
    completeness_criteria,
    f"metrics.llm_as_judge.direct.criterias.{completeness_criteria.name}",
    overwrite=True,
)


# now = define the judge metric using the criteria
adherence_metric = LLMJudgeDirect(
    inference_engine=CrossProviderInferenceEngine(  # or your favorite inference model
        model="llama-3-3-70b-instruct", max_tokens=1024
    ),
    criteria=adherence_criteria,
    # the fields from the generation task to be presented to the judge. Those fields must be present
    # in the generation task so they can be embedded here
    context_fields={
        "question": "question",
        "instructions": "metadata/template/instruction",
    },
    criteria_field="criteria",
    generate_summaries=False,
    check_positional_bias=False,
)
add_to_catalog(
    adherence_metric,
    "metrics.rag.response_generation.adherence_with_format.llama_3_3_70b_instruct_judge",
    overwrite=True,
)

# now = define the judge metric using the criteria
completeness_metric = LLMJudgeDirect(
    inference_engine=CrossProviderInferenceEngine(  # or your favorite inference model
        model="llama-3-3-70b-instruct", max_tokens=1024
    ),
    criteria=completeness_criteria,
    # the fields from the generation task to be presented to the judge. Those fields must be present
    # in the generation task so they can be embedded here
    context_fields={"question": "question", "reference_answers": "reference_answers"},
    criteria_field="criteria",
    generate_summaries=False,
    check_positional_bias=False,
)

add_to_catalog(
    completeness_metric,
    "metrics.rag.response_generation.answer_completeness.llama_3_3_70b_instruct_judge",
    overwrite=True,
)
