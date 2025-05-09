from unitxt.catalog import add_to_catalog
from unitxt.metrics import ToolCallingMetric, ToolCallKeyValueExtraction

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
    ToolCallKeyValueExtraction(__description__ = """Metric that evaluates tool call predictions with reference calls.
First generate unique key value pairs for the tool name, and all the parameters (including nested parameter).
Reports average accuracy for each key, as well as micro and macro averages across all keys.

Supports only a single reference call per prediction.

""", metric="metrics.accuracy"),
    "metrics.tool_calling.key_value.accuracy",
    overwrite=True,
)

add_to_catalog(
    ToolCallKeyValueExtraction(__description__ = """Metric that evaluates tool call predictions with reference calls.
First generate unique key value pairs for the tool name, and all the parameters (including nested parameter).
Supports only a single reference call per prediction.

Reports average token_overlap for each key, as well as micro and macro averages across all keys.
""", metric="metrics.token_overlap",score_prefix="token_overlap_"),
    "metrics.tool_calling.key_value.token_overlap",
    overwrite=True,
)

