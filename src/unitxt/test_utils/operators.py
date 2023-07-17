from ..operator import StreamingOperator
from ..stream import MultiStream
from ..type_utils import is_typing_type
from typing import List
import json

def apply_operator(operator:StreamingOperator, inputs: List[dict]):
    multi_stream = MultiStream({'test': inputs})
    output_multi_stream = operator(multi_stream)
    output_stream = output_multi_stream['test']
    return list(output_stream)
    
def test_operator(operator:StreamingOperator, inputs: List[dict], targets: List[dict]):
    
    assert is_typing_type(operator, StreamingOperator), "operator must be an Operator"
    assert is_typing_type(inputs, List[dict]), "inputs must be a list of dicts"
    assert is_typing_type(outputs, List[dict]), "outputs must be a list of dicts"
    
    outputs = apply_operator(operator, inputs)
    
    for input, output in zip(outputs, targets):
        assert json.dumps(input, sort_keys=True) == json.dumps(output, sort_keys=True), "input and output must be equal"
    
    return True
    
    
    
    
    
    
    