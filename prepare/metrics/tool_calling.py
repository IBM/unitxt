from unitxt.catalog import add_to_catalog
from unitxt.metrics import (
    MultiTurnToolCallingMetric,
    ReflectionToolCallingMetric,
    ReflectionToolCallingMetricSyntactic,
    ToolCallingMetric,
    ToolCallKeyValueExtraction,
)

add_to_catalog(
    ToolCallingMetric(
        __description__="""Metric that evaluates tool call predictions with reference calls.
Generate aggregated metrics on tool name, tool parameter selection , and tool parameter value type.
Can supports multiple references."""
    ),
    "metrics.tool_calling",
    overwrite=True,
)

add_to_catalog(
    ToolCallKeyValueExtraction(
        __description__="""Metric that evaluates tool call predictions with reference calls.
First generate unique key value pairs for the tool name, and all the parameters (including nested parameter).
Reports average accuracy for each key, as well as micro and macro averages across all keys.

Supports only a single reference call per prediction.

""",
        metric="metrics.accuracy",
    ),
    "metrics.tool_calling.key_value.accuracy",
    overwrite=True,
)

add_to_catalog(
    ToolCallKeyValueExtraction(
        __description__="""Metric that evaluates tool call predictions with reference calls.
First generate unique key value pairs for the tool name, and all the parameters (including nested parameter).
Supports only a single reference call per prediction.

Reports average token_overlap for each key, as well as micro and macro averages across all keys.
""",
        metric="metrics.token_overlap",
        score_prefix="token_overlap_",
    ),
    "metrics.tool_calling.key_value.token_overlap",
    overwrite=True,
)

add_to_catalog(
    MultiTurnToolCallingMetric(
        __description__="""A metric that assesses tool call predictions for their conformity to the tool schema."""
    ),
    "metrics.tool_calling.multi_turn.validity",
    overwrite=True,
)

add_to_catalog(
    ReflectionToolCallingMetric(
        __description__="""A metric that assesses tool call predictions for both syntactic correctness and semantic validity, using predefined checks combined with LLM-based evaluations. For each instance, it returns a score reflecting its overall validity, as well as a breakdown of the specific checks/metrics that passed or failed, including hallucination check, value format alignment, function selection and agentic constraints satisfaction. Each metric also contains an evidence from the input, an explanation describing the reflection decision, a confidence, and a validity score with a range of 1-5 (higher score -> more valid)."""
    ),
    "metrics.tool_calling.reflection",
    overwrite=True,
)

add_to_catalog(
    ReflectionToolCallingMetricSyntactic(
        __description__="""This metric evaluates whether a model's tool call outputs are structurally valid by checking their compliance with the provided tool schema. For each instance, it returns a binary score (True for valid, False for invalid), and aggregates these into a global percentage across all instances. The evaluation covers a wide range of possible issues, including nonexistent functions or parameters, incorrect parameter types, missing required parameters, values outside allowed ranges, JSON schema violations, invalid or empty API specifications, and malformed tool calls. The main reported score, overall_valid (aliased as score), reflects the proportion of calls that are fully valid, making the metric a measure of syntactic and schema-level correctness rather than semantic accuracy. Each metric also contains an explanation describing the errors that it detected (if no errors were found - the explanation will be None)."""
    ),
    "metrics.tool_calling.reflection.syntactic",
    overwrite=True,
)
