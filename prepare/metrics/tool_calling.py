from unitxt.catalog import add_to_catalog
from unitxt.metrics import ToolCallingMetric

add_to_catalog(
    ToolCallingMetric(
        __description__="""Metric for tool calling"""
    ),
    "metrics.tool_calling",
    overwrite=True,
)
