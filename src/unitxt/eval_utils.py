from functools import singledispatch
from typing import List

import pandas as pd

from .operator import SequentialOperator
from .stream import MultiStream

# use singledispatch to overload the function evaluate(), supporting both dataframe input and list input


@singledispatch
def evaluate(dataset, metric_names: List[str]):
    pass


@evaluate.register
def _(dataset: list, metric_names: List[str]):
    for metric_name in metric_names:
        multi_stream = MultiStream.from_iterables({"test": dataset}, copying=True)
        metrics_operator = SequentialOperator(steps=[metric_name])
        instances = list(metrics_operator(multi_stream)["test"])
        for entry, instance in zip(dataset, instances):
            entry[metric_name] = instance["score"]["instance"]["score"]
    return dataset


@evaluate.register
def _(dataset: pd.DataFrame, metric_names: List[str]):
    return pd.DataFrame(evaluate(dataset.to_dict("records"), metric_names=metric_names))
