from ..metric import Metric
from ..stream import MultiStream, Stream
from ..type_utils import is_typing_type
from typing import List
import json

def apply_metric(metric:Metric, predictions: List[str], references: List[List[str]]):
    
    assert is_typing_type(metric, Metric), "operator must be an Operator"
    assert is_typing_type(predictions, List[str]), "predictions must be a list of strings"
    assert is_typing_type(references, List[List[str]]), "references must be a list of lists of strings"
    
    test_iterable = [{'prediction': prediction, 'references': reference} for prediction, reference in zip(predictions, references)]
    multi_stream = MultiStream.from_iterables({'test': test_iterable})
    output_multi_stream = metric(multi_stream)
    output_stream = output_multi_stream['test']
    return list(output_stream)
    
def test_metric(metric:Metric, inputs: List[dict], targets: List[dict]):
    
    assert is_typing_type(metric, Metric), "operator must be an Operator"
    assert is_typing_type(inputs, List[dict]), "inputs must be a list of dicts"
    assert is_typing_type(outputs, List[dict]), "outputs must be a list of dicts"
    
    outputs = apply_metric(metric, inputs)
    
    for input, output in zip(outputs, targets):
        assert json.dumps(input, sort_keys=True) == json.dumps(output, sort_keys=True), "input and output must be equal"
    
    return True
    