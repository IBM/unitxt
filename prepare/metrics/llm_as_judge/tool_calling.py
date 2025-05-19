from unitxt import add_to_catalog
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.llm_as_judge import LLMJudgeDirect
from unitxt.llm_as_judge_constants import CriteriaWithOptions
from unitxt.types import ToolCall

option_map = {
    "Excellent": 1.0,
    "Good": 0.75,
    "Mediocre": 0.5,
    "Bad": 0.25,
    "Very Bad": 0.0,
}

tool_calling_criteria = CriteriaWithOptions.from_obj(
    {
        "name": "tool_calling_correctness",
        "description": "The response correctly uses tool calls as expected, including the right tool names and parameters, in line with the reference or user query and instructions.",
        "options": [
            {
                "name": "Excellent",
                "description": "All tool calls are correct, including names and parameters, matching the reference or user expectations precisely.",
            },
            {
                "name": "Good",
                "description": "Tool calls are mostly correct with minor errors that do not affect the functionality or intent.",
            },
            {
                "name": "Mediocre",
                "description": "The response attempts tool calls with partial correctness, but has notable issues in tool names, structure, or parameters.",
            },
            {
                "name": "Bad",
                "description": "The tool calling logic is largely incorrect, with significant mistakes in tool usage or missing key calls.",
            },
            {
                "name": "Very Bad",
                "description": "The tool calls are completely incorrect, irrelevant, or missing when clearly required.",
            },
        ],
        "option_map": option_map,
    }
)

add_to_catalog(
    tool_calling_criteria,
    "metrics.llm_as_judge.direct.criteria.tool_calling_correctness",
    overwrite=True,
)

tool_calling_metric = LLMJudgeDirect(
    inference_engine=CrossProviderInferenceEngine(
        model="llama-3-3-70b-instruct", max_tokens=1024, temperature=0, provider="watsonx"
    ),
    criteria=tool_calling_criteria,
    context_fields={
        "tools": "tools",
        "reference_tool_calls": "reference_calls",
        "user_query": "query",
    },
    criteria_field="criteria",
    generate_summaries=False,
    check_positional_bias=False,
    prediction_type=ToolCall,
)

add_to_catalog(
    tool_calling_metric,
    "metrics.tool_calling.correctness.llama_3_3_70b_instruct_judge",
    overwrite=True,
)
