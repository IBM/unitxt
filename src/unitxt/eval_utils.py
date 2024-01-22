import sys
from ast import literal_eval
from typing import List

import pandas as pd

from .operator import SequentialOperator
from .stream import MultiStream


def evaluate(dataset: pd.DataFrame, metric_names: List[str]):
    result = dataset.copy()
    # prepare the input stream
    for metric_name in metric_names:
        multi_stream = MultiStream.from_iterables(
            {"test": dataset.to_dict("records")}, copying=True
        )
        metrics_operator = SequentialOperator(steps=[metric_name])
        instances = list(metrics_operator(multi_stream)["test"])
        result[metric_name] = [
            instance["score"]["instance"]["score"] for instance in instances
        ]
    return result


if __name__ == "__main__":
    df = pd.read_csv(
        filepath_or_buffer=sys.argv[1],
        converters={
            "ground_truths": literal_eval,
            "ground_truths_context_ids": literal_eval,
            "contexts": literal_eval,
            "context_ids": literal_eval,
        },
    )

    evaluate(
        df,
        metric_names=[
            "metrics.rag.mrr",
            "metrics.rag.map",
            "metrics.rag.answer_correctness",
            "metrics.rag.context_relevance",
            "metrics.rag.faithfulness",
            "metrics.rag.answer_relevance",
            "metrics.rag.context_perplexity",
        ],
    ).round(2).to_csv("dataset_out.csv")
