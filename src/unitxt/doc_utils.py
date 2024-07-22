DOCUMENTATION_URL = "https://www.unitxt.ai/en/latest/"
DOCUMENTATION_HUGGINGFACE_METRICS = "docs/adding_metric.html#adding-a-hugginface-metric"
DOCUMENTATION_ADDING_TASK = "docs/adding_task.html"


def additional_info(path: str) -> str:
    return f"\\nFor more information: see {DOCUMENTATION_URL}/{path} \\n"
