from collections import defaultdict
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
        exclude_scores = {"score", "score_name"}
        for instance in instances:
            score_values = defaultdict(list)
            for score_name, score_value in instance["score"]["instance"].items():
                if score_name not in exclude_scores:
                    score_values[score_name].append(score_value)
            for score_name, score_value_list in score_values.items():
                result[f"{metric_name}.{score_name}"] = score_value_list
        result[metric_name] = [
            instance["score"]["instance"]["score"] for instance in instances
        ]
    return result
