from ..operator import StreamingOperator
from ..stream import MultiStream
from ..type_utils import is_typing_type
from typing import List

def test_operator(operator:StreamingOperator, inputs: List[dict], outputs: List[dict]):
    
    assert is_typing_type(operator, StreamingOperator), "operator must be an Operator"
    assert is_typing_type(inputs, List[dict]), "inputs must be a list of dicts"
    assert is_typing_type(outputs, List[dict]), "outputs must be a list of dicts"
    
    multi_stream = MultiStream({'test': inputs})
    output_multi_stream = operator(multi_stream)
    output_stream = output_multi_stream['test']
    
    for input, output in zip(output_stream, outputs):
        for key in output:
            assert key in input, f"key {key} not found in input"
            assert input[key] == output[key], f"key {key} does not match"
    
    return True
    
    
    
    
    
    
    