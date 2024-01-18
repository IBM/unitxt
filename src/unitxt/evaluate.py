from ast import literal_eval
from typing import List

import pandas as pd

from unitxt.operator import SequentialOperator
from unitxt.stream import MultiStream


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
        "/Users/matanorbach/Documents/matan/src/fm_eval_workspace/rag_metrics/datasets/RAG/rag_dataset.csv",
        converters={
            "ground_truths": literal_eval,
            "ground_truths_context_ids": literal_eval,
            "contexts": literal_eval,
            "context_ids": literal_eval,
        },
    )

    evaluate(df, metric_names=["metrics.rag.mrr", "metrics.rag.context_relevancy"])
