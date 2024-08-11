from ..logging_utils import get_logger
from ..stream import MultiStream

logger = get_logger()
rag_fields = {"ground_truths", "answer", "contexts", "question"}

test_examples = [
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports a variety of foundation models",
        "contexts": ["Supported foundation models available with watsonx.ai"],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports no foundation models",
        "contexts": ["Supported foundation models available with watsonx.ai"],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
    },
]


def test_metric(metric_pipeline):
    metric_name = metric_pipeline.metric.main_score
    multi_stream = MultiStream.from_iterables({"test": test_examples}, copying=True)
    metric_outputs = list(metric_pipeline(multi_stream)["test"])
    metric_preds = [ex["score"]["instance"][metric_name] for ex in metric_outputs]
    logger.info(f"======\n{metric_name}:")
    for i in range(len(metric_outputs)):
        logger.info(f"{i}) Input: {metric_outputs[i]['task_data']}: {metric_preds[i]}")
