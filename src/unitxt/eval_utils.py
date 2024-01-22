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
